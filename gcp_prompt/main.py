import json
import os
import functions_framework
from flask import request, jsonify, session
import ast
import re
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple
import traceback
import requests
import math
from datetime import datetime, timedelta
from google.oauth2 import service_account
from googleapiclient.discovery import build
from openai import OpenAI
import google.auth
from google.cloud import secretmanager
import hashlib
import io

def access_secret(project_id, secret_id, version_id="latest"):
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")



def fetch_json(url):
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def round_down(n, decimals=1):
    multiplier = 10 ** decimals
    return math.floor(n * multiplier) / multiplier

    
def load_teacher_prompts() -> List[Dict[str, Any]]:
    """Fetch the teacher's prompt from local answer_prompts.json file."""
    try:
        # Get the directory where the Cloud Function code is deployed
        current_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(current_dir, 'answer_prompts.json')
        
        # Read the JSON file
        with open(json_path, 'r', encoding='utf-8') as f:
            prompts = json.load(f)
        
        print(f"Loaded {len(prompts)} teacher prompts")
        return prompts

    except Exception as e:
        print(f"Error fetching teacher prompt: {str(e)}")
        return None



FORBIDDEN_KEYWORDS = ["eval", "exec", "os", "sys", "subprocess"]
users_url = "https://raw.githubusercontent.com/alexlopespereira/ipynb-autograde/refs/heads/master/data/users.json"
courses_url = "https://raw.githubusercontent.com/alexlopespereira/ipynb-autograde/refs/heads/master/data/courses.json"
deadlines_url = "https://raw.githubusercontent.com/alexlopespereira/ipynb-autograde/refs/heads/master/data/deadlines.json"  # Update with actual URL


users_data = fetch_json(users_url)
courses_data = fetch_json(courses_url)
deadlines_data = fetch_json(deadlines_url)

teacher_prompts = load_teacher_prompts()

OPENAI_GPT_MODEL = os.environ.get("OPENAI_GPT_MODEL")
os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

llm_provider = os.environ.get("LLM_PROVIDER", "OPENAI").upper()

openai_client = OpenAI(api_key=access_secret(
            project_id="autograde-314802",
            secret_id="OPENAI_API_KEY"
        ))

deepseek_client = OpenAI(
    api_key=access_secret(
            project_id="autograde-314802",
            secret_id="DEEPSEEK_API_KEY"
        ),
    base_url="https://api.deepseek.com/v1"
)

def get_teacher_prompt(function_id: str) -> str:
    """Fetch the teacher's prompt from local answer_prompts.json file."""
    # Find matching prompt
    print(f"Function ID: {function_id}")
    for prompt_data in teacher_prompts:
        print(f"Prompt data: {prompt_data}")
        if prompt_data["function_id"] == function_id:
            print(f"Found matching prompt: {prompt_data}")
            return prompt_data["prompt"], prompt_data["code"]
    return None


def get_reflection_history(course: str, user_email: str) -> List[str]:
    """Retrieve previous passing reflection answers from Google Sheets for a specific user."""
    try:
        # Get credentials
        submission_credentials = json.loads(access_secret(
            project_id="autograde-314802",
            secret_id="GOOGLE_SUBMISSION_CREDENTIALS"
        ))

        SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
        SPREADSHEET_ID = '1IwvQoqdMUklaw5P2CZH7YWKdeZhhehJljd1TdI0RDP0'
        RANGE_NAME = 'records!A:L'  # All columns

        # Load credentials from service account file
        creds = service_account.Credentials.from_service_account_info(
            info=submission_credentials,
            scopes=SCOPES
        )

        # Build the Sheets API service
        service = build('sheets', 'v4', credentials=creds)

        # Get all rows
        result = service.spreadsheets().values().get(
            spreadsheetId=SPREADSHEET_ID,
            range=RANGE_NAME
        ).execute()
        
        rows = result.get('values', [])
        
        # Filter rows for passing reflection answers
        # Column indices: B=email, C=course, D=class_number, E=exercise_number, G=passed, L=reflection_text
        passing_reflections = []
        for row in rows:
            if len(row) >= 12:  # Ensure row has all needed columns
                row_email = row[1]
                row_course = row[2]
                row_exercise = row[4]
                row_passed = row[6].lower() == 'true'
                reflection_text = row[11]
                
                # Check if this is a reflection exercise for the specified course and user
                if (row_course == course and 
                    row_email == user_email and
                    'R' in row_exercise and 
                    row_passed and 
                    reflection_text.strip()):
                    passing_reflections.append(reflection_text)
        #print(f"passing_reflections={passing_reflections[:3]}")
        return passing_reflections

    except Exception as e:
        print(f"Error retrieving reflection history: {str(e)}")
        return []



def prompt_completion(user_prompt, is_reflection=False, course=None, class_number=None, user_email=None, provider="OPENAI"):
    """Generate code or evaluate reflection using specified LLM API."""
    
    
    if is_reflection:
        # Get previous passing reflections
        previous_reflections = []
        if course and user_email:
            previous_reflections = get_reflection_history(course, user_email)
        
        previous_reflections_text = "\n\n".join([
            f"Previous passing reflection #{i+1}:\n{text}"
            for i, text in enumerate(previous_reflections)
        ])
        
        content = f"""You are a teaching assistant evaluating a student's reflection. 
        The student was asked to summarize what they learned in this class.
        
        Evaluate the following reflection, considering:
        - Depth of understanding
        - Specific concepts mentioned
        - Connection between ideas
        - Personal insights
        - Originality (the answer should not be too similar to the student's previous passing answers)
        
        Here are the student's previous passing reflections for reference:
        {previous_reflections_text}
        
        Return a JSON with two fields:
        - "passed": boolean indicating if the reflection meets quality standards AND is sufficiently original
        - "feedback": brief explanation of the evaluation, including any concerns about similarity to previous answers
        
        Current student reflection to evaluate:
        {user_prompt}"""
    else:
        content = f"In your answer do not return in hypertext format, return only raw text. Do not produce code for importing packages, all the allowed packages are already imported. Do not create code for testing the function. Write a Python function for the following prompt:\n{user_prompt}"
    client = openai_client if provider == "OPENAI" else deepseek_client
    try:
        response = client.chat.completions.create(
            model=OPENAI_GPT_MODEL if provider == "OPENAI" else "deepseek-reasoner",
            messages=[{"role": "user", "content": content}],
            max_completion_tokens=2500
        )
        generated_response = response.choices[0].message.content.strip()
        
        if not generated_response:
            print("prompt completion with o1-mini failed, retrying with gpt-4o-mini")
            response = client.chat.completions.create(
                model="gpt-5",
                messages=[{"role": "user", "content": content}],
                max_completion_tokens=2500
            )
            generated_response = response.choices[0].message.content.strip()
            
    except Exception as e:
        print(f"{provider} API error: {str(e)}")
        raise

    # Remove any markdown code block formatting
    generated_response = generated_response.replace("```", "")
    
    if not generated_response:
        raise Exception("The generated response is empty. You probably sent a too large prompt.")
    
    return generated_response


def hash_email(email: str, salt: str) -> str:
    """Hash email with provided salt using SHA-256."""
    return hashlib.md5(f"{email}{salt}".encode()).hexdigest()

def log_to_sheets(row_data):
    """
    Logs a row of data to Google Sheets
    """
    # Get salt from environment variable
    salt = access_secret(project_id="autograde-314802", secret_id="EMAIL_HASH_SALT")
    if not salt:
        print("Warning: EMAIL_HASH_SALT not set, using empty salt")
    
    # Add hashed email as the last column
    row_data.append(hash_email(row_data[1], salt))  # row_data[1] contains the email

    submission_credentials = json.loads(access_secret(
        project_id="autograde-314802",
        secret_id="GOOGLE_SUBMISSION_CREDENTIALS"
    ))

    SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
    SPREADSHEET_ID = '1IwvQoqdMUklaw5P2CZH7YWKdeZhhehJljd1TdI0RDP0'
    RANGE_NAME = 'records!A:M'  # Updated to include the new hashed email column

    try:
        # Load credentials from service account file
        creds = service_account.Credentials.from_service_account_info(
           info=submission_credentials,
           scopes=SCOPES
        )

        # Build the Sheets API service
        service = build('sheets', 'v4', credentials=creds)

        # Prepare the request body
        body = {
            'values': [row_data]
        }

        # Append the row to the sheet
        service.spreadsheets().values().append(
            spreadsheetId=SPREADSHEET_ID,
            range=RANGE_NAME,
            valueInputOption='RAW',
            insertDataOption='INSERT_ROWS',
            body=body
        ).execute()

    except Exception as e:
        print(f"Error logging to sheets: {str(e)}")


def check_deadline(function_id, submission_time, course):
    """Check if submission is within deadline"""
    try:
        # Extract class day from function_id (e.g., "A2" from "A2_E6")
        class_day = function_id.split("-")[0]
        
        # Get deadlines for the specific course
        course_deadlines = deadlines_data.get(course, {}).get("deadlines", [])
        
        deadline_info = next(
            (d for d in course_deadlines if d["class_day"] == class_day),
            None
        )
        print(f"class_day={class_day}, deadline_info={deadline_info}")
        
        if not deadline_info:
            return True, f"No deadline specified for {course} - {class_day}"
            
        # Parse timezone offset
        timezone_str = deadline_info.get("timezone", "UTC-0")
        timezone_offset = int(timezone_str.replace("UTC", ""))
        
        # Convert deadline to UTC
        deadline = datetime.fromisoformat(deadline_info["deadline"])
        deadline = deadline + timedelta(hours=timezone_offset)  # Convert to UTC
        
        # Convert submission time to UTC (assuming it's already in UTC)
        submission = datetime.fromisoformat(submission_time)
        
        return submission <= deadline, f"{deadline_info['deadline']} {timezone_str}"
    except Exception as e:
        print(f"Error checking deadline: {e}")
        return True, "Error checking deadline"



def validate_dataframe(df: pd.DataFrame, expected_format: dict) -> Tuple[bool, List[str], dict]:
    """
    Validate DataFrame against expected format specifications.
    Returns:
        Tuple containing:
        - bool: Whether validation passed
        - List[str]: List of error messages if validation failed, empty list otherwise
        - dict: Summary of verified content if validation passed, empty dict if failed
    """
    if not isinstance(df, pd.DataFrame):
        return False, ["Input must be a pandas DataFrame"], {}

    if not isinstance(expected_format, dict) or 'data' not in expected_format:
        return False, ["Invalid expected_format structure"], {}

    data_spec = expected_format
    validation_rules = data_spec.get('validation_rules', {})
    errors = []

    # Initialize summary dictionary (only used if validation passes)
    summary = {
        "verified_samples": [],
        "aggregation_summary": {},
        "column_analysis": {
            "expected_columns": [],
            "provided_columns": list(df.columns),
            "missing_columns": [],
            "unexpected_columns": [],
            "possible_matches": {}
        }
    }

    # Enhanced column validation
    if 'columns' in data_spec['data']:
        expected_cols = data_spec['data']['columns']
        summary["column_analysis"]["expected_columns"] = expected_cols

        # Find missing and unexpected columns
        missing_cols = set(expected_cols) - set(df.columns)
        unexpected_cols = set(df.columns) - set(expected_cols)

        # Only treat missing columns as errors
        if missing_cols:
            errors.append(f"Missing required columns: {', '.join(missing_cols)}")

            # Look for possible matches between missing and unexpected columns
            if unexpected_cols:
                for unexpected_col in unexpected_cols:
                    for expected_col in missing_cols:
                        if len(set(unexpected_col.lower()) & set(expected_col.lower())) > len(expected_col) / 2:
                            errors.append(f"Column '{unexpected_col}' might be a misspelling of required column '{expected_col}'")

        # Record unexpected columns in summary but don't treat them as errors
        if unexpected_cols:
            summary["column_analysis"]["unexpected_columns"] = list(unexpected_cols)
            summary["column_analysis"]["missing_columns"] = list(missing_cols)

    # If we already have errors, return early
    if errors:
        return False, errors, {}

    # Validate dtypes for existing columns
    if 'dtypes' in data_spec['data']:
        for col, expected_dtype in data_spec['data']['dtypes'].items():
            if col in df.columns:
                if str(df[col].dtype) != expected_dtype:
                    errors.append(f"Column '{col}' has incorrect dtype. Expected {expected_dtype}, got {df[col].dtype}")

    # If we have dtype errors, return early
    if errors:
        return False, errors, {}

    # Validate aggregation checks with tolerance
    if 'aggregation_checks' in data_spec['data']:
        agg_checks = data_spec['data']['aggregation_checks']
        round_decimals = validation_rules.get('round_decimals')
        agg_summary = {}
        tolerance = validation_rules.get('tolerance', 3)  # Default tolerance of 3 units

        # Check total rows
        if 'total_rows' in agg_checks:
            row_check = agg_checks['total_rows']
            total_rows = len(df)
            agg_summary['total_rows'] = row_check
            if 'min' in row_check and total_rows < row_check['min']:
                errors.append(f"DataFrame has {total_rows} rows, minimum required is {row_check['min']}")
            if 'max' in row_check and total_rows > row_check['max']:
                errors.append(f"DataFrame has {total_rows} rows, maximum allowed is {row_check['max']}")

        # Check sums with tolerance
        if 'sum' in agg_checks and not errors:
            agg_summary['sums'] = agg_checks['sum']
            for col, expected_sum in agg_checks['sum'].items():
                actual_sum = round_down(df[col].sum(), round_decimals)
                if abs(actual_sum - expected_sum) > tolerance:
                    errors.append(f"Sum mismatch for column {col}. Expected {expected_sum}, got {actual_sum} (tolerance: ±{tolerance})")

        # Check means with tolerance
        if 'mean' in agg_checks and not errors:
            agg_summary['means'] = agg_checks['mean']
            for col, expected_mean in agg_checks['mean'].items():
                actual_mean = round_down(df[col].mean(), round_decimals)
                if abs(actual_mean - expected_mean) > tolerance:
                    errors.append(f"Mean mismatch for column {col}. Expected {expected_mean}, got {actual_mean} (tolerance: ±{tolerance})")

        summary['aggregation_summary'] = agg_summary

    # If we have aggregation errors, return early
    if errors:
        return False, errors, {}

    # Rest of the validation code remains the same...
    # Validate sample rows
    if 'sample_rows' in data_spec['data']:
        threshold = validation_rules.get('row_match_threshold', 0.001)
        for i, sample in enumerate(data_spec['data']['sample_rows'], 1):
            sample_summary = {
                "filter": sample['filter'],
                "expected_values": sample['expected_values']
            }
            summary['verified_samples'].append(sample_summary)

            filters = []
            for col, value in sample['filter'].items():
                filters.append(df[col] == value)
            filtered_df = df[pd.concat(filters, axis=1).all(axis=1)]

            if filtered_df.empty:
                errors.append(f"Sample {i}: No rows match filter {sample['filter']} and expected values {sample['expected_values']}")
                return False, errors, {}

            for _, row in filtered_df.iterrows():
                for col, expected_val in sample['expected_values'].items():
                    actual_val = row[col]
                    if isinstance(expected_val, (int, float)) and isinstance(actual_val, (int, float)):
                        if abs(float(actual_val) - float(expected_val)) > threshold:
                            errors.append(f"Sample {i}: Value mismatch for column {col}. Expected {expected_val}, got {actual_val}")
                            return False, errors, {}

    # Only return the full summary if there are no errors
    if errors:
        return False, errors, {}

    return True, [], summary

class TestCaseValidator:
    TEST_CASES_URL = "https://raw.githubusercontent.com/alexlopespereira/ipynb-autograde/refs/heads/master/data/questions.json"

    def __init__(self, function_id: str, course: str):
        """Initialize validator with function_id and fetch relevant test cases."""
        self.function_id = function_id
        self.course = course
        self.test_cases = self._fetch_test_cases()

    def _fetch_test_cases(self) -> List[Dict[str, Any]]:
        """Fetch and filter test cases for the specified function_id and course."""
        try:
            response = requests.get(self.TEST_CASES_URL)
            response.raise_for_status()
            all_courses_test_cases = response.json()
            
            # Select test cases for the specific course
            course_test_cases = all_courses_test_cases.get(self.course, [])
            
            # Filter by function_id
            return [tc for tc in course_test_cases if tc.get("function_id") == self.function_id]
        except Exception as e:
            print(f"Error fetching test cases: {str(e)}")
            return []

    def validate_type(self, value: Any, expected_type: str) -> bool:
        """Validate if the value matches the expected type."""
        type_mapping = {
            "str": str,
            "list": list,
            "int": int,
            "float": (float, int),  # Accept both float and int for float type
            "dict": dict,
            "any": object
        }
        return isinstance(value, type_mapping.get(expected_type, object))

    def validate_requirements(self, code: str, requirements: List[str]) -> bool:
        """Validate if code meets all specified requirements."""
        requirement_checks = {
            "numpy array": lambda c: "np.array" in c or "numpy.array" in c,
            "pandas series": lambda c: "pd.Series" in c or "pandas.Series" in c,
            "pandas dataframe": lambda c: "pd.DataFrame" in c or "pandas.DataFrame" in c
        }

        for req in requirements:
            req_lower = req.lower()
            for key, check_func in requirement_checks.items():
                if key in req_lower and not check_func(code):
                    return False
        return True

    def _round_floats(self, value: Any, places: int = 1) -> Any:
        """Round floating point numbers in various data structures."""
        if isinstance(value, (float, int)):
            return round(float(value), places)
        elif isinstance(value, list):
            return [self._round_floats(x, places) for x in value]
        elif isinstance(value, dict):
            return {k: self._round_floats(v, places) for k, v in value.items()}
        return value

    def _normalize_text(self, text: str) -> str:
        """Normalize text by removing extra whitespace and converting to lowercase."""
        return re.sub(r'\s+', ' ', text.strip().lower())

    def _compare_lists(self, actual: List, expected: List, tolerance: float = 0.1) -> bool:
        """Compare lists with special handling for numeric values."""
        if len(actual) != len(expected):
            return False

        for a, e in zip(actual, expected):
            if isinstance(e, (int, float)) and isinstance(a, (int, float)):
                if abs(float(a) - float(e)) > tolerance:
                    return False
            elif isinstance(e, list) and isinstance(a, list):
                if not self._compare_lists(a, e, tolerance):
                    return False
            elif isinstance(e, dict) and isinstance(a, dict):
                if not self._compare_dicts(a, e, tolerance):
                    return False
            elif str(a) != str(e):  # Convert to string for non-numeric comparisons
                return False
        return True

    def _compare_dicts(self, actual: Dict, expected: Dict, tolerance: float = 0.1) -> bool:
        """Compare dictionaries with special handling for numeric values."""
        if set(actual.keys()) != set(expected.keys()):
            return False

        for key in expected:
            if isinstance(expected[key], (int, float)) and isinstance(actual[key], (int, float)):
                if abs(float(actual[key]) - float(expected[key])) > tolerance:
                    return False
            elif isinstance(expected[key], list) and isinstance(actual[key], list):
                if not self._compare_lists(actual[key], expected[key], tolerance):
                    return False
            elif isinstance(expected[key], dict) and isinstance(actual[key], dict):
                if not self._compare_dicts(actual[key], expected[key], tolerance):
                    return False
            elif str(actual[key]) != str(expected[key]):
                return False
        return True

    def compare_outputs(self, actual, expected, output_type=None, tolerance=0.1):
        """Compare actual and expected outputs with DataFrame handling."""
        # Handle DataFrame conversion
        if isinstance(actual, pd.DataFrame):
            is_valid, errors, summary = validate_dataframe(actual, expected)
            return is_valid, summary, errors

        # Existing comparison logic
        if isinstance(expected, dict) and isinstance(actual, dict):
            return self._compare_dicts(actual, expected, tolerance), expected, []
        elif isinstance(expected, list) and isinstance(actual, list):
            return self._compare_lists(actual, expected, tolerance), expected, []

        # Handle None values
        if actual is None and expected is None:
            return True, expected, []
        if actual is None or expected is None:
            return False, expected, []

        # Validate output type if specified
        if output_type and not self.validate_type(actual, output_type):
            return False, expected, []

        # Handle text pattern matching (for exercises like A2-E7 to A2-E11)
        if isinstance(expected, str) and isinstance(actual, str):
            if '\n' in expected or '=' in expected or 'o' in expected:
                return self._normalize_text(actual) == self._normalize_text(expected), expected, []

        # Handle numeric comparisons
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            return abs(float(actual) - float(expected)) <= tolerance, expected, []

        # Handle list comparisons
        if isinstance(expected, list) and isinstance(actual, list):
            return self._compare_lists(actual, expected, tolerance), expected, []

        # Handle dictionary comparisons
        if isinstance(expected, dict) and isinstance(actual, dict):
            return self._compare_dicts(actual, expected, tolerance), expected, []

        # Handle pandas DataFrame/Series comparisons
        if isinstance(expected, (pd.DataFrame, pd.Series)) and isinstance(actual, (pd.DataFrame, pd.Series)):
            return actual.equals(expected), expected, []

        # Handle numpy array comparisons
        if isinstance(expected, np.ndarray) and isinstance(actual, np.ndarray):
            return np.allclose(actual, expected, rtol=tolerance), expected, []

        # Default string comparison
        return str(actual) == str(expected), expected, []

    def _convert_to_serializable(self, obj):
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, dict):
            # Check for integer keys
            if any(isinstance(k, int) for k in obj.keys()):
                raise ValueError("Integer keys are not accepted in dictionaries. All keys must be strings.")
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (set, tuple)):
            raise ValueError("Tuples and sets cannot be serialized. Please explicitly convert to a list before returning.")
        return obj


    def run_validation(self, code: str) -> Dict[str, Any]:
        """Run all test cases for the provided code and return results."""
        if not self.test_cases:
            return {
                "status": "error",
                "message": f"No test cases found for function_id: {self.function_id}"
            }

        try:
            # Clean and prepare code
            patterns = [
                r'^\s*import\s+.*?(?=\n|$)',  # Matches import statements
                r'^\s*from\s+.*?import\s+.*?(?=\n|$)'  # Matches from ... import statements
            ]
            cleaned_text = code
            for pattern in patterns:
                cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.MULTILINE)

            full_code = f"""
import pandas as pd
from pandas import date_range
import numpy as np
import requests
import math
import io
from io import BytesIO
from io import StringIO
import random
import zipfile
from datetime import date, timedelta
import datetime
import re
from collections import defaultdict
{cleaned_text}
            """

            # Execute code and get function
            exec_globals = {}
            exec(full_code, exec_globals)
            func_name = [name for name in exec_globals if callable(exec_globals[name])][-1]
            func = exec_globals[func_name]

            # Run all test cases
            test_results = []
            for test_case in self.test_cases:
                try:
                    inputs = test_case.get("input", [])
                    if not isinstance(inputs, list):
                        inputs = [inputs]

                    # Execute function and compare results
                    #print(f"inputs: {inputs}")
                    if "input_transformation" in test_case and test_case["input_transformation"]['input_type'] == "dataframe":
                        df = pd.DataFrame(inputs[test_case["input_transformation"]['replace_index']])
                        inputs[0] = df
                    result = func(*inputs)
                    passed, summary, errors = self.compare_outputs(
                        result,
                        test_case['expected'],
                        test_case.get("output_type")
                    )

                    if isinstance(result, pd.DataFrame) and test_case["expected"]["type"] == "dataframe":
                        test_results.append({
                            "testcase_id": test_case["testcase_id"],
                            "passed": passed,
                            "expected": test_case["expected"],
                            "actual": self._convert_to_serializable(summary),
                            "error": errors
                        })
                    else:
                        test_results.append({
                            "testcase_id": test_case["testcase_id"],
                            "passed": passed,
                            "expected": test_case["expected"],
                            "actual": self._convert_to_serializable(result),
                            "error": None
                        })

                except Exception as e:
                    print(f"Error while running your function: {str(e)}")
                    test_results.append({
                        "testcase_id": test_case["testcase_id"],
                        "passed": False,
                        "expected": self._convert_to_serializable(test_case["expected"]),
                        "actual": None,
                        "error": f"Error while running your function: {str(e)}"
                    })

            # Calculate summary statistics
            total_tests = len(test_results)
            passed_tests = sum(1 for r in test_results if r["passed"])

            return {
                "status": "success",
                "code": cleaned_text,
                "test_results": test_results,
                "summary": {
                    "total_tests": total_tests,
                    "passed_tests": passed_tests,
                    "success_rate": f"{(passed_tests / total_tests) * 100:.2f}%"
                }
            }

        except Exception as e:
            return {
                "status": "error",
                "message": "An error occurred while processing the code",
                "error": str(e),
                "traceback": traceback.format_exc()
            }


def call_python(data, course: str):
    """Cloud Function to execute and validate Python code against test cases."""
    # data = request.get_json()
    code = data.get("code")
    function_id = data.get("function_id")

    if not all([code, function_id]):
        return {
            "status": "error",
            "message": "Missing required parameters: code or function_id"
        }

    # Initialize validator and run validation with timeout handling
    validator = TestCaseValidator(function_id, course)
    print(f"Validator: {validator}")    
    result = validator.run_validation(code)
    print(f"Result: {result}")

    if result["status"] == "error":
        return result

    return result


def analyze_code_safety(code):
    """Analyze code for potential security issues."""
    try:
        formatted_code = code.replace("\\n", "\n")
        tree = ast.parse(formatted_code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in FORBIDDEN_KEYWORDS:
                    return False, f"Use of '{node.func.id}' is not allowed."
        return True, None
    except SyntaxError as e:
        if "unterminated string literal" in str(e):
            return True, None
        return False, f"Syntax error in code: {e}"
    except Exception as e:
        print(f"An unexpected error occurred in analyze_code_safety: {e}")
        return False, f"An unexpected error occurred during code analysis: {e}"


def get_prompt_feedback(student_prompt: str, teacher_prompt: str, student_code: str, teacher_code: str, passed: bool, provider="OPENAI") -> str:
    """Generate constructive feedback by comparing student and teacher prompts."""
    
    content = f"""You are a helpful teaching assistant. Compare the following two prompts and provide constructive feedback:

Teacher's prompt: {teacher_prompt}
Student's prompt: {student_prompt}

code generated with student's prompt: {student_code}
code generated with teacher's prompt: {teacher_code}

Passed: {passed}

If the student did not pass, analyze the differences and provide feedback that:
1. Does NOT reveal the actual solution or teacher's prompt
2. Asks guiding questions about missing key elements
3. Points out areas where clarity could be improved
4. Points out areas where the code generated with student's prompt is not correct
Choose two of the above criteria to return as feedback to the student. Choose the two criteria causes the most differences between the student's and teacher's code.

If the student did pass, analyze the differences and provide feedback that:
1. Reminds about good prompt writing practices
2. Points out areas where clarity could be improved

Do not greet the user the student. Do not write motivational messages at the end. Return only the feedback, written in a supportive tone. Be concise. Always answer in brazilian Portuguese."""
    client = openai_client if provider == "OPENAI" else deepseek_client
    try:
        response = client.chat.completions.create(
            model=OPENAI_GPT_MODEL if provider == "OPENAI" else "deepseek-reasoner",
            messages=[{"role": "user", "content": content}],
            max_completion_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating prompt feedback: {str(e)}")
        return None

def get_google_credentials():
    """Fetch Google service account credentials from Secret Manager."""
    client = secretmanager.SecretManagerServiceClient()
    name = "projects/ipynb-autograde/secrets/GOOGLE_SUBMISSION_CREDENTIALS/versions/latest"
    response = client.access_secret_version(request={"name": name})
    return json.loads(response.payload.data.decode("UTF-8"))


@functions_framework.http
def validate_code(request):
    """Cloud function to validate student code."""
    # Set CORS headers for the preflight request
    """ if request.method == 'OPTIONS':
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST',
            'Access-Control-Allow-Headers': 'Content-Type,Authorization',
            'Access-Control-Max-Age': '3600'
        }
        return ('', 204, headers)

    # Set CORS headers for the main request
    headers = {
        'Access-Control-Allow-Origin': '*'
    } """

    try:
        valid_courses = courses_data.get("courses", [])
        authorized_users = set()

        for course in valid_courses:
            authorized_users.update(users_data.get(course, []))

        AUTHORIZED_USERS = authorized_users

    except Exception as e:
        print(f"An error occurred: {e}")

    print(f"Authorized users: {AUTHORIZED_USERS}")

    """ auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return jsonify({"error": "Missing or invalid Authorization header"}), 401

    token = auth_header.split("Bearer ")[1]

    try:
        # Verify Google token
        response = requests.get(f"https://oauth2.googleapis.com/tokeninfo?access_token={token}")
        if response.status_code != 200:
            return jsonify({"error": "Invalid token"}), 403

        token_info = response.json()
        email = token_info.get("email")

        if not email:
            return jsonify({"error": "Email not found in token"}), 403

        if email not in AUTHORIZED_USERS:  # Changed from using session to direct check
            return jsonify({"error": "Unauthorized user"}), 403


    except Exception as e:
        print(f"Token validation error: {e}")
        return jsonify({"error": "Invalid token"}), 403
 """
    try:
        data = request.get_json()
        user_prompt = data.get("prompt")
        function_id = data.get("function_id")
        user_email = data.get("user_email")
        course = data.get("course")

        print(f"User email: {user_email}")
        print(f"User prompt: {user_prompt}")
        print(f"Function ID: {function_id}")
        print(f"Course: {course}")

        if not all([user_prompt, function_id, user_email, course]):
            return jsonify({
                "status": "error",
                "message": "Missing required parameters: prompt, user_email, course or function_id"
            }), 400
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": "An error occurred while processing the request",
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500
    

    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    submission_id = f"{user_email}_{function_id}"
    within_deadline, deadline = check_deadline(function_id, timestamp, course)
    class_number, exercise_number = function_id.split("-")

    try:
        # Check if this is a reflection question
        is_reflection = "-R" in function_id
        if is_reflection:
            print("Reflection question")
            # Handle reflection submission
            evaluation = prompt_completion(
                user_prompt, 
                is_reflection=True,
                course=course,
                class_number=class_number,
                user_email=user_email
            )
            evaluation = re.sub(r"^json\s*", "", evaluation)
            try:
                evaluation_dict = json.loads(evaluation)
                result = {
                    "passed": evaluation_dict["passed"],
                    "feedback": evaluation_dict["feedback"],
                    "reflection_text": user_prompt
                }
            except json.JSONDecodeError:
                return jsonify({"error": "Invalid evaluation format from AI"}), 500

            
            print(f"Evaluation result: {result}")

            # Log reflection to sheets with different column structure
            log_to_sheets([
                timestamp,
                user_email,
                course,
                class_number,
                exercise_number,
                submission_id,
                str(result["passed"]),
                str(within_deadline),  # Get deadline status based on class day
                deadline,  # Get deadline timestamp based on class day
                result["feedback"],
                f"{user_email}_{function_id}_{timestamp}",
                user_prompt  # Store the actual reflection text
            ])

            return jsonify(result)

        else:
            # Get teacher's prompt and generate feedback
            #print("Not reflection question")
            # Generate code from student's prompt
            generated_code = prompt_completion(user_prompt, user_email=user_email, provider="OPENAI")
            generated_code = re.sub(r"^python\s*", "", generated_code)
            
            # Validate request data
            is_safe, error_message = analyze_code_safety(generated_code)
            if not is_safe:
                return jsonify({"error": f"Unsafe code: {error_message}"}), 400
            
            # Return cloud function response
            print(f"Generated code: {generated_code}")
            result = call_python({"code": generated_code, "function_id": function_id}, course=course)
            print(f"Result: {result}")
            error_message = result.get("error", None)
            print(f"Error message: {error_message}")
            passed = False
            if "test_results" in result and result["test_results"]:
                passed = all(test.get("passed", False) for test in result["test_results"])
            print(f"Passed: {passed}")
            # Get feedback comparing both prompts and their generated code
            #if not passed:
            teacher_prompt, teacher_code = get_teacher_prompt(function_id)
            print(f"Teacher prompt: {teacher_prompt}")
            print(f"Teacher code: {teacher_code}")
            prompt_feedback = get_prompt_feedback(user_prompt, teacher_prompt, generated_code, teacher_code, passed, provider="OPENAI")
            #else:
            #    prompt_feedback = None
            print(f"Prompt feedback: {prompt_feedback}")
            result.update({
                "user_email": user_email,
                "function_id": function_id,
                "prompt_feedback": prompt_feedback,
                "generated_code": generated_code
            })
            #print("Result:", result)
            # Add deadline check
            log_to_sheets([
                timestamp,
                user_email,
                course,
                class_number,
                exercise_number,
                submission_id,
                str(passed),
                str(within_deadline),  # Add deadline status
                deadline,              # Add deadline timestamp
                error_message or "None",
                f"{user_email}_{function_id}_{timestamp}",
                user_prompt
            ])
            
            return jsonify(result)

    except requests.exceptions.Timeout:
        return jsonify({
            "error": "Request timed out. The operation took too long to complete.",
            "user_email": user_email,
            "function_id": function_id
        }), 504  # Gateway Timeout status code
    except Exception as e:
        return jsonify({
            "error": f"Error processing request: {str(e)}",
            "user_email": user_email,
            "function_id": function_id
        }), 500
