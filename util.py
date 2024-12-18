import pandas as pd
import ast
import subprocess
import time

# Test case definitions
test_cases = [
    {"input": (3, 5), "expected": 8, "id": "AAA"},
    {"input": (10, 20), "expected": 30, "id": "AAA"},
    {"input": (-1, 1), "expected": 0, "id": "AAA"},
]

# Forbidden keywords and constructs
FORBIDDEN_KEYWORDS = ["import", "open", "eval", "exec", "os", "sys", "subprocess"]

def analyze_code_safety(code):
    """
    Analyzes the code for forbidden keywords or constructs.
    Args:
        code (str): The code submitted by the user.
    Returns:
        tuple: (is_safe, error_message)
    """
    try:
        # Parse the code into an AST (Abstract Syntax Tree)
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                return False, "Imports are not allowed."
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in FORBIDDEN_KEYWORDS:
                    return False, f"Use of '{node.func.id}' is not allowed."
        return True, None
    except SyntaxError as e:
        return False, f"Syntax error in code: {e}"

def execute_code_with_timeout(code, inputs, timeout=5):
    """
    Executes the user code in a subprocess with a timeout.
    Args:
        code (str): The user code.
        inputs (tuple): Input arguments for the function.
        timeout (int): Time limit for execution in seconds.
    Returns:
        dict: The execution result.
    """
    start_time = time.time()
    try:
        exec_globals = {}
        exec(code, exec_globals)
        func_name = [name for name in exec_globals if callable(exec_globals[name])][-1]
        func = exec_globals[func_name]
        output = func(*inputs)
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout:
            return {"status": "timeout", "error": "Execution exceeded time limit."}
        return {"status": "success", "output": output}
    except Exception as e:
        return {"status": "error", "error": str(e)}


# Function to dynamically test student implementations
def test_student_code(implementation_text, test_id):
    """
    Tests a function provided as text against predefined test cases.

    Args:
        function_name (str): The name of the function to test.
        implementation_text (str): The student's function implementation as text.

    Returns:
        dict: A dictionary containing the test results.
    """
    results = {"test_results": []}
    try:
        # Execute the implementation text
        exec(implementation_text, globals())

        # Extract the first function defined in the implementation_text
        function_name = [name for name in globals() if callable(globals()[name]) and name not in dir(__builtins__)][-1]
        function_to_test = globals()[function_name]
        results["function_name"] = function_name

    except Exception as e:
        results["error"] = f"Error in student implementation: {e}"
        return results

    # Execute the test cases
    for i, case in enumerate(test_cases):
        if case.get("id") != test_id:
            continue  # Skip test cases that don't match the given ID

        inputs = case["input"]
        expected = case["expected"]

        try:
            # Call the function dynamically
            output = function_to_test(*inputs)
            if output == expected:
                results["test_results"].append({"test_case": i + 1, "status": "passed"})
            else:
                results["test_results"].append({
                    "test_case": i + 1,
                    "status": "failed",
                    "output": output,
                    "expected": expected,
                })
        except Exception as e:
            results["test_results"].append({
                "test_case": i + 1,
                "status": "error",
                "error": str(e),
            })

    # Check if all tests passed
    results["all_tests_passed"] = all(r["status"] == "passed" for r in results["test_results"])
    return results

