from flask import jsonify
import re
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Union, Optional
import traceback
import requests
from decimal import Decimal
import json
from functools import lru_cache

class TestCaseValidator:
    def __init__(self, test_cases: List[Dict[str, Any]]):
        self.test_cases = test_cases
        
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

    def validate_test_case(self, test_case: Dict[str, Any], code: str, func: callable) -> Dict[str, Any]:
        """Validate a single test case and return the result."""
        try:
            # Check requirements if specified
            requirements = test_case.get("requirements", [])
            if requirements and not self.validate_requirements(code, requirements):
                return {
                    "testcase_id": test_case["testcase_id"],
                    "passed": False,
                    "expected": test_case["expected"],
                    "actual": None,
                    "error": "Code does not meet requirements"
                }

            # Prepare inputs
            inputs = test_case.get("input", [])
            if not isinstance(inputs, list):
                inputs = [inputs]

            # Validate input types if specified
            input_types = test_case.get("input_type", {})
            for i, input_val in enumerate(inputs, 1):
                if str(i) in input_types:
                    if not self.validate_type(input_val, input_types[str(i)]):
                        return {
                            "testcase_id": test_case["testcase_id"],
                            "passed": False,
                            "expected": test_case["expected"],
                            "actual": None,
                            "error": f"Input {i} has incorrect type"
                        }

            # Execute function
            result = func(*inputs)

            # Compare output
            passed = self.compare_outputs(
                result,
                test_case["expected"],
                test_case.get("output_type")
            )

            return {
                "testcase_id": test_case["testcase_id"],
                "passed": passed,
                "expected": test_case["expected"],
                "actual": result,
                "error": None
            }

        except Exception as e:
            return {
                "testcase_id": test_case["testcase_id"],
                "passed": False,
                "expected": test_case["expected"],
                "actual": None,
                "error": str(e)
            }

@lru_cache(maxsize=1)
def fetch_test_cases(url: str) -> List[Dict[str, Any]]:
    """Fetch test cases from URL with caching."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching test cases: {str(e)}")
        return []

def filter_test_cases(test_cases: List[Dict[str, Any]], function_id: str) -> List[Dict[str, Any]]:
    """Filter test cases for specific function_id."""
    return [tc for tc in test_cases if tc.get("function_id") == function_id]

def call_python(request):
    """Cloud Function to execute and validate Python code against test cases."""
    try:
        # Get request data
        data = request.get_json()
        code = data.get("code")
        function_id = data.get("function_id")
        test_cases_url = data.get("test_cases_url", "https://raw.githubusercontent.com/your-repo/test-cases.json")

        if not all([code, function_id]):
            return jsonify({
                "status": "error",
                "message": "Missing required parameters: code or function_id"
            }), 400

        # Fetch and filter test cases
        all_test_cases = fetch_test_cases(test_cases_url)
        test_cases = filter_test_cases(all_test_cases, function_id)

        if not test_cases:
            return jsonify({
                "status": "error",
                "message": f"No test cases found for function_id: {function_id}"
            }), 404

        # Initialize validator
        validator = TestCaseValidator(test_cases)

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
        for test_case in test_cases:
            result = validator.validate_test_case(test_case, cleaned_text, func)
            test_results.append(result)

        # Calculate summary statistics
        total_tests = len(test_results)
        passed_tests = sum(1 for r in test_results if r["passed"])

        return jsonify({
            "status": "success",
            "code": cleaned_text,
            "test_results": test_results,
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "success_rate": f"{(passed_tests/total_tests)*100:.2f}%"
            }
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": "An error occurred while processing the request",
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500