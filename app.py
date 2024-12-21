import json
import os
import re
from flask import Flask, request, jsonify, redirect, session, url_for
import ast
from openai import OpenAI
from flask.json.provider import JSONProvider
import aiohttp
import asyncio
from google.oauth2 import service_account
from google.auth.transport.requests import Request
from google.oauth2.id_token import verify_oauth2_token
from google_auth_oauthlib.flow import Flow

class UTF8JSONProvider(JSONProvider):
    def dumps(self, obj, **kwargs):
        kwargs.setdefault('ensure_ascii', False)
        return json.dumps(obj, **kwargs)

    def loads(self, s, **kwargs):
        try:
            if isinstance(s, bytes):
                s = s.decode('utf-8')
            return json.loads(s, **kwargs)
        except UnicodeDecodeError:
            return json.loads(s.decode('latin-1'), **kwargs)


app = Flask(__name__)
app.json_provider_class = UTF8JSONProvider
app.json = UTF8JSONProvider(app)
app.secret_key = os.environ.get("FLASK_SECRET_KEY")
DEBUG = os.environ.get("DEBUG", None) == "True"

os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"  # For local testing, disable HTTPS requirement
# CLIENT_SECRETS_FILE = "client_secret_oauth.json"  # Downloaded JSON file from Google Console
credentials = json.loads(os.environ.get('GOOGLE_CREDENTIALS'))
AUTHORIZED_USERS = {"alexlopespereira@gmail.com", "alex.pereira.tablet@gmail.com"}
flow = Flow.from_client_config(
    credentials,
    scopes=["https://www.googleapis.com/auth/userinfo.email"],
    redirect_uri="https://seal-app-pmncf.ondigitalocean.app/callback"
)

test_cases = [
    {"input": [3, 5], "expected": 8, "function_id": "A1-E2", "testcase_id": "BBB"},
    {"input": (10, 20), "expected": 30, "function_id": "A1-E2", "testcase_id": "BBC"},
    {"input": (-1, 1), "expected": 0, "function_id": "A1-E2", "testcase_id": "BBD"},
]

FORBIDDEN_KEYWORDS = ["import", "open", "eval", "exec", "os", "sys", "subprocess"]

@app.route("/login")
def login():
    """Redirect the user to the Google OAuth login page."""
    authorization_url, _ = flow.authorization_url()
    return redirect(authorization_url)

@app.route("/callback")
def callback():
    """Handle the OAuth callback and validate the user."""
    print("test")
    print(f"Callback hit: {request.url}")
    print(f"Query Parameters: {request.args}")
    # flow.fetch_token(authorization_response=request.url)
    return jsonify({"message": f"Welcome"})
    #
    # # Verify the ID token
    # credentials = flow.credentials
    # id_token = credentials.id_token
    # info = verify_oauth2_token(id_token, Request())
    #
    # # Extract email and validate against authorized users
    # email = info.get("email")
    # if email in AUTHORIZED_USERS:
    #     session["user_email"] = email
    #     return jsonify({"message": f"Welcome, {email}!"})
    # else:
    #     return jsonify({"error": "Unauthorized user"}), 403

def google_cloud_function_mockup(payload):
    from main import call_python
    response = call_python(payload, DEBUG=True)
    result = response[0].get_json()
    result['error'] = ''
    return result


def prompt_completion(user_prompt):
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates Python functions."},
            {"role": "user",
             "content": f"In your answer return only the python code, and no text before neither after the code. Write a Python function for the following prompt:\n{user_prompt}"}
        ],
        max_tokens=150
    )
    generated_code = response.choices[0].message.content.strip().replace("```","")
    generated_code = re.sub(r"^python\s*", "", generated_code)
    return generated_code


def analyze_code_safety(code):
    try:
        formatted_code = code.replace("\\n", "\n")
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


async def execute_test_case(session, cloud_function_url, headers, generated_code, test_case):
    if DEBUG:
        cloud_result = google_cloud_function_mockup({"code": generated_code, "inputs": test_case["input"]})
    else:
        async with session.post(
                cloud_function_url,
                json={"code": generated_code, "inputs": test_case["input"]},
                headers=headers
        ) as cloud_response:
            if cloud_response.status != 200:
                return {
                    "testcase_id": test_case["testcase_id"],
                    "input": test_case["input"],
                    "expected": test_case["expected"],
                    "actual": None,
                    "passed": False,
                    "error": f"Error from cloud function: {await cloud_response.text()}"
                }
            cloud_result = await cloud_response.json()

    actual_output = cloud_result.get("output")
    return {
        "testcase_id": test_case["testcase_id"],
        "input": test_case["input"],
        "expected": test_case["expected"],
        "actual": actual_output,
        "passed": actual_output == test_case["expected"],
        "error": cloud_result.get("error", "")
    }


@app.route('/api/validate', methods=['POST'])
async def validate_student_code():
    from flask import session
    if "user_email" not in session:
        return redirect(url_for("login"))
    try:
        data = request.get_json()
        if not data or "prompt" not in data or "function_id" not in data or "user_email" not in data:
            return jsonify({"error": "Invalid request format"}), 400

        user_prompt = data["prompt"]
        function_id = data["function_id"]
        user_email = data["user_email"]

        generated_code = prompt_completion(user_prompt)
        is_safe, error_message = analyze_code_safety(generated_code)
        if not is_safe:
            return jsonify({"error": f"Unsafe code: {error_message}"}), 400

        relevant_test_cases = [tc for tc in test_cases if tc["function_id"] == function_id]
        if not relevant_test_cases:
            return jsonify({"error": "No test cases found for this function_id"}), 404


        cloud_function_url = os.environ.get("GCP_FUNC_URL")
        credentials = service_account.IDTokenCredentials.from_service_account_file(
            './key.json', target_audience=cloud_function_url
        )
        request_adapter = Request()
        credentials.refresh(request_adapter)
        headers = {
            'Authorization': f'Bearer {credentials.token}',
            'Content-Type': 'application/json'
        }

        test_results = []
        async with aiohttp.ClientSession() as session:
            tasks = [
                execute_test_case(session, cloud_function_url, headers, generated_code, test_case)
                for test_case in relevant_test_cases
            ]
            test_results = await asyncio.gather(*tasks)

        passed_count = sum(1 for result in test_results if result["passed"])

        result = {
            "function_id": function_id,
            "user_email": user_email,
            "code": generated_code,
            "test_results": test_results,
            "total_tests": len(test_results),
            "passed_tests": passed_count,
            "success_rate": passed_count / len(test_results) if test_results else 0
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {e}"}), 500


if __name__ == '__main__':
    app.run(debug=True)
