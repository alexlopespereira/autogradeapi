from flask import jsonify
import re
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Union, Optional, Tuple
import traceback
import requests
from decimal import Decimal
import math

def round_down(n, decimals=1):
    multiplier = 10 ** decimals
    return math.floor(n * multiplier) / multiplier


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

    def __init__(self, function_id: str):
        """Initialize validator with function_id and fetch relevant test cases."""
        self.function_id = function_id
        self.test_cases = self._fetch_test_cases()

    def _fetch_test_cases(self) -> List[Dict[str, Any]]:
        """Fetch and filter test cases for the specified function_id."""
        try:
            response = requests.get(self.TEST_CASES_URL)
            response.raise_for_status()
            all_test_cases = response.json()
            return [tc for tc in all_test_cases if tc.get("function_id") == self.function_id]
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
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (set, tuple)):
            return list(obj)
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
import numpy as np
import requests
import math
import io
from io import BytesIO
import random
import zipfile
import datetime
from datetime import date, timedelta
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
                    print(f"inputs: {inputs}")
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
                            "expected": test_case['expected'],
                            "actual": summary,
                            "error": errors
                        })
                    else:
                        test_results.append({
                            "testcase_id": test_case["testcase_id"],
                            "passed": passed,
                            "expected": test_case["expected"],
                            "actual": result,
                            "error": None
                        })

                except Exception as e:
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


def call_python(request):
    """Cloud Function to execute and validate Python code against test cases."""
    try:
        data = request.get_json()
        code = data.get("code")
        function_id = data.get("function_id")

        if not all([code, function_id]):
            return jsonify({
                "status": "error",
                "message": "Missing required parameters: code or function_id"
            }), 400

        # Initialize validator and run validation
        validator = TestCaseValidator(function_id)
        result = validator.run_validation(code)

        if result["status"] == "error":
            return jsonify(result), 500

        return jsonify(result), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": "An error occurred while processing the request",
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500