from flask import jsonify
import re
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Union, Optional, Tuple
import traceback
import requests
from decimal import Decimal


class TestCaseValidator:
    TEST_CASES_URL = "https://raw.githubusercontent.com/alexlopespereira/ipynb-autograde/refs/heads/master/data/questions.json"

    def _validate_aggregates(self, df: pd.DataFrame, agg_checks: Dict, validation_rules: Dict = None) -> Tuple[bool, List[str]]:
        """
        Validate DataFrame aggregate values.

        Args:
            df: DataFrame to validate
            agg_checks: Dictionary containing aggregation checks
            validation_rules: Dictionary containing validation rules including round_decimals

        Returns:
            Tuple of (bool, List[str]) indicating success and any error messages
        """
        errors = []
        round_decimals = validation_rules.get('round_decimals') if validation_rules else None

        # Check total rows
        if "total_rows" in agg_checks:
            row_check = agg_checks["total_rows"]
            total_rows = len(df)
            if "min" in row_check and total_rows < row_check["min"]:
                errors.append(f"DataFrame has {total_rows} rows, minimum required is {row_check['min']}")
            if "max" in row_check and total_rows > row_check["max"]:
                errors.append(f"DataFrame has {total_rows} rows, maximum allowed is {row_check['max']}")

        # Check column sums
        if "sum" in agg_checks:
            for col, expected_sum in agg_checks["sum"].items():
                if col not in df.columns:
                    errors.append(f"Column {col} not found for sum check")
                    continue

                actual_sum = df[col].sum()
                if round_decimals is not None:
                    actual_sum = round(actual_sum, round_decimals)
                    # Ensure expected_sum has the same precision
                    expected_sum = round(float(expected_sum), round_decimals)

                if actual_sum != expected_sum:
                    errors.append(f"Sum mismatch for column {col}. Expected {expected_sum}, got {actual_sum}")

        # Check column means
        if "mean" in agg_checks:
            for col, expected_mean in agg_checks["mean"].items():
                if col not in df.columns:
                    errors.append(f"Column {col} not found for mean check")
                    continue

                actual_mean = df[col].mean()
                if round_decimals is not None:
                    actual_mean = round(actual_mean, round_decimals)
                    # Ensure expected_mean has the same precision
                    expected_mean = round(float(expected_mean), round_decimals)

                if actual_mean != expected_sum:
                    errors.append(f"Mean mismatch for column {col}. Expected {expected_mean}, got {actual_mean}")

        return len(errors) == 0, errors

    def _match_pattern(self, value: str, pattern: str) -> bool:
        """Match string value against a regex pattern."""
        try:
            return bool(re.match(pattern, str(value)))
        except re.error:
            return False

    def _check_value_range(self, value: Any, range_spec: Dict) -> bool:
        """Check if value falls within specified range."""
        if not isinstance(value, (int, float)):
            return False

        if "min" in range_spec and value < range_spec["min"]:
            return False
        if "max" in range_spec and value > range_spec["max"]:
            return False
        return True

    def _filter_dataframe_row(self, df: pd.DataFrame, filter_conditions: Dict) -> pd.DataFrame:
        """Filter DataFrame based on given conditions including pattern matching."""
        query_parts = []

        for col, condition in filter_conditions.items():
            if isinstance(condition, dict):
                if "pattern" in condition:
                    mask = df[col].apply(lambda x: self._match_pattern(x, condition["pattern"]))
                    df = df[mask]
                elif "min" in condition or "max" in condition:
                    if "min" in condition:
                        df = df[df[col] >= condition["min"]]
                    if "max" in condition:
                        df = df[df[col] <= condition["max"]]
            else:
                query_parts.append(f'{col} == @condition')

        if query_parts:
            query = ' & '.join(query_parts)
            return df.query(query)
        return df

    def _compare_row_values(self, row: pd.Series, expected_values: Dict,
                            threshold: float = 0.001) -> bool:
        """Compare a single row with expected values including range checks."""
        for col, expected_val in expected_values.items():
            if col not in row:
                return False

            actual_val = row[col]

            # Handle range specifications
            if isinstance(expected_val, dict) and ("min" in expected_val or "max" in expected_val):
                if not self._check_value_range(actual_val, expected_val):
                    return False
            # Handle numeric comparisons
            elif isinstance(expected_val, (int, float)) and isinstance(actual_val, (int, float)):
                if abs(float(actual_val) - float(expected_val)) > threshold:
                    return False
            # Handle string comparisons
            elif str(actual_val) != str(expected_val):
                return False
        return True

    def _compare_dataframes(self, actual: pd.DataFrame, expected_dict: Dict,
                          validation_rules: Dict = None) -> Tuple[bool, List[str]]:
        """Compare DataFrame with enhanced validation including aggregates and patterns."""
        if validation_rules is None:
            validation_rules = {}

        errors = []

        try:
            # Check columns
            missing_cols = set(expected_dict["columns"]) - set(actual.columns)
            if missing_cols:
                errors.append(f"Missing columns: {missing_cols}")
                return False, errors

            # Check data types if specified
            if "dtypes" in expected_dict:
                for col, dtype in expected_dict["dtypes"].items():
                    if str(actual[col].dtype) != dtype:
                        errors.append(f"Column {col} has wrong dtype. Expected {dtype}, got {actual[col].dtype}")
                        return False, errors

            # Round decimals if specified
            if validation_rules.get("round_decimals") is not None:
                actual = actual.round(validation_rules["round_decimals"])

            # Check aggregate values if specified
            if "aggregation_checks" in expected_dict:
                is_valid, agg_errors = self._validate_aggregates(actual, expected_dict["aggregation_checks"], validation_rules)
                if not is_valid:
                    errors.extend(agg_errors)
                    return False, errors


            # Check sample rows
            threshold = validation_rules.get("row_match_threshold", 0.001)

            for i, sample in enumerate(expected_dict["sample_rows"], 1):
                # Filter the DataFrame for matching rows
                filtered_df = self._filter_dataframe_row(actual, sample["filter"])

                if filtered_df.empty:
                    errors.append(f"Sample {i}: No rows match filter {sample['filter']}")
                    return False, errors

                # Check if any row matches the expected values
                found_match = False
                for _, row in filtered_df.iterrows():
                    if self._compare_row_values(row, sample["expected_values"], threshold):
                        found_match = True
                        break

                if not found_match:
                    errors.append(
                        f"Sample {i}: No matching values found for filter {sample['filter']}. "
                        f"Expected values: {sample['expected_values']}"
                    )
                    return False, errors

            return True, []

        except Exception as e:
            errors.append(f"Error comparing DataFrames: {str(e)}")
            return False, errors

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

    def compare_outputs(self, actual: Any, expected: Any, output_type: Optional[str] = None,
                        tolerance: float = 0.1) -> bool:
        """Compare actual and expected outputs with comprehensive type checking and comparison."""
        # Handle None values
        if actual is None and expected is None:
            return True
        if actual is None or expected is None:
            return False

        # Validate output type if specified
        if output_type and not self.validate_type(actual, output_type):
            return False

        # Handle text pattern matching (for exercises like A2-E7 to A2-E11)
        if isinstance(expected, str) and isinstance(actual, str):
            if '\n' in expected or '=' in expected or 'o' in expected:
                return self._normalize_text(actual) == self._normalize_text(expected)

        # Handle numeric comparisons
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            return abs(float(actual) - float(expected)) <= tolerance

        # Handle list comparisons
        if isinstance(expected, list) and isinstance(actual, list):
            return self._compare_lists(actual, expected, tolerance)

        # Handle dictionary comparisons
        if isinstance(expected, dict) and isinstance(actual, dict):
            return self._compare_dicts(actual, expected, tolerance)

        # Handle pandas DataFrame/Series comparisons
        if isinstance(expected, (pd.DataFrame, pd.Series)) and isinstance(actual, (pd.DataFrame, pd.Series)):
            return actual.equals(expected)

        # Handle numpy array comparisons
        if isinstance(expected, np.ndarray) and isinstance(actual, np.ndarray):
            return np.allclose(actual, expected, rtol=tolerance)

        # Default string comparison
        return str(actual) == str(expected)

    def run_validation(self, code: str) -> Dict[str, Any]:
        """Run all test cases for the provided code and return results."""
        if not self.test_cases:
            return {
                "status": "error",
                "message": f"No test cases found for function_id: {self.function_id}"
            }

        try:
            # Clean and prepare code
            pattern = r'^(?:from\s+\w+(?:\.\w+)*\s+import\s+(?:\w+(?:\s*,\s*\w+)*|\*)|import\s+(?:\w+(?:\s*,\s*\w+)*|\w+(?:\.\w+)*))(?:\s+as\s+\w+)?$'
            cleaned_text = re.sub(pattern, '', code, flags=re.MULTILINE)
            full_code = f"""
import pandas as pd
import numpy as np
import requests
import io
import random
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
                    # Prepare inputs
                    inputs = test_case.get("input", [])
                    if not isinstance(inputs, list):
                        inputs = [inputs]

                    # Check requirements if specified
                    requirements = test_case.get("requirements", [])
                    if requirements and not self.validate_requirements(cleaned_text, requirements):
                        test_results.append({
                            "testcase_id": test_case["testcase_id"],
                            "passed": False,
                            "expected": test_case["expected"],
                            "actual": None,
                            "error": "Code does not meet requirements"
                        })
                        continue

                    # Execute function and compare results
                    result = func(*inputs)
                    passed = self.compare_outputs(
                        result,
                        test_case["expected"],
                        test_case.get("output_type")
                    )

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
                        "expected": test_case["expected"],
                        "actual": None,
                        "error": str(e)
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