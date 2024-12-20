from flask import Flask, request, jsonify
import ast
import time
import aiohttp
import asyncio
from google.auth.transport.requests import Request
from google.oauth2 import service_account

app = Flask(__name__)

# Test case definitions
test_cases = [
    {"input": (3, 5), "expected": 8, "function_id": "AAA", "testcase_id": "BBB"},
    {"input": (10, 20), "expected": 30, "function_id": "AAA", "testcase_id": "BBC"},
    {"input": (-1, 1), "expected": 0, "function_id": "AAA", "testcase_id": "BBD"},
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
        # Replace escaped newlines with actual newlines
        formatted_code = code.replace("\\n", "\n")
        # Parse the code into an AST (Abstract Syntax Tree)
        tree = ast.parse(formatted_code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                return False, "Imports are not allowed."
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in FORBIDDEN_KEYWORDS:
                    return False, f"Use of '{node.func.id}' is not allowed."
        return True, None
    except SyntaxError as e:
        return False, f"Syntax error in code: {e}"

@app.route('/api/validate', methods=['POST'])
async def validate_student_code():
    """
    Flask API endpoint to validate a student's function implementation.

    Expects a JSON payload with:
    - implementation (str): The function implementation as a string.
    - function_id (str): The ID of the function to validate against.
    - testcase_id (str): The ID of the specific test case to validate.
    - user_email (str): The email of the user submitting the implementation.

    Returns:
        JSON response with validation results.
    """
    try:
        # Parse the JSON payload
        data = request.get_json()

        if not data or "implementation" not in data or "function_id" not in data or "testcase_id" not in data or "user_email" not in data:
            return jsonify({"error": "Invalid request format. 'implementation', 'function_id', 'testcase_id', and 'user_email' are required."}), 400

        implementation_text = data["implementation"]
        function_id = data["function_id"]
        testcase_id = data["testcase_id"]
        user_email = data["user_email"]

        # Analyze code safety
        is_safe, error_message = analyze_code_safety(implementation_text)
        if not is_safe:
            return jsonify({"error": f"Unsafe code: {error_message}"}), 400

        # Find the specific test case
        test_case = next((tc for tc in test_cases if tc["function_id"] == function_id and tc["testcase_id"] == testcase_id), None)
        if not test_case:
            return jsonify({"error": f"No test case found for function_id: {function_id} and testcase_id: {testcase_id}"}), 404

        # Set up the authenticated call
        cloud_function_url = "https://us-central1-autograde-314802.cloudfunctions.net/pandas-gcp-test"
        credentials = service_account.IDTokenCredentials.from_service_account_file(
            './key.json', target_audience=cloud_function_url
        )
        request_adapter = Request()
        credentials.refresh(request_adapter)

        headers = {
            'Authorization': f'Bearer {credentials.token}',
            'Content-Type': 'application/json'
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(cloud_function_url, json={"code": implementation_text, "inputs": test_case["input"], "user_email": user_email}, headers=headers) as cloud_response:
                if cloud_response.status != 200:
                    return jsonify({"error": f"Error from cloud function: {await cloud_response.text()}"}), cloud_response.status

                cloud_result = await cloud_response.json()

        result = {
            "function_id": function_id,
            "testcase_id": testcase_id,
            "status": "passed" if cloud_result["status"] == "success" and cloud_result["output"] == test_case["expected"] else "failed",
            "output": cloud_result.get("output"),
            "expected": test_case["expected"],
            "error": cloud_result.get("error"),
            "user_email": user_email
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
