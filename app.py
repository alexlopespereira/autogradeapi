import json
import os
import re
import requests
from flask import Flask, request, jsonify, session
import ast
from openai import OpenAI
from flask.json.provider import JSONProvider
from google.oauth2 import service_account
from google.auth.transport.requests import Request
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

users_url = "https://raw.githubusercontent.com/alexlopespereira/ipynb-autograde/refs/heads/master/data/users.json"
courses_url = "https://raw.githubusercontent.com/alexlopespereira/ipynb-autograde/refs/heads/master/data/courses.json"

def fetch_json(url):
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

try:
    users_data = fetch_json(users_url)
    courses_data = fetch_json(courses_url)

    valid_courses = courses_data.get("courses", [])
    authorized_users = set()

    for course in valid_courses:
        authorized_users.update(users_data.get(course, []))

    AUTHORIZED_USERS = authorized_users
    print("AUTHORIZED_USERS updated successfully:", AUTHORIZED_USERS)

except Exception as e:
    print(f"An error occurred: {e}")

app = Flask(__name__)
app.json_provider_class = UTF8JSONProvider
app.json = UTF8JSONProvider(app)
app.secret_key = os.environ.get("SECRET_KEY")
DEBUG = os.environ.get("DEBUG", None) == "True"
OPENAI_GPT_MODEL = os.environ.get("OPENAI_GPT_MODEL")
os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

credentials = json.loads(os.environ.get('GOOGLE_CREDENTIALS'))
flow = Flow.from_client_config(
    credentials,
    scopes=["https://www.googleapis.com/auth/userinfo.email"],
    redirect_uri="https://seal-app-pmncf.ondigitalocean.app/callback"
)

FORBIDDEN_KEYWORDS = ["eval", "exec", "os", "sys", "subprocess"]

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
            pass
        return False, f"Syntax error in code: {e}"

def prompt_completion(user_prompt):
    """Generate code using OpenAI API based on user prompt."""
    client = OpenAI()
    content = f"In your answer do not return in hypertext format, return only raw text. Do not produce code for importing packages, all the allowed packages are already imported. Write a Python function for the following prompt:\n{user_prompt}"
    
    response = client.chat.completions.create(
        model=OPENAI_GPT_MODEL,
        messages=[{
            "role": "user",
            "content": content
        }],
        max_completion_tokens=2500
    )
    
    generated_code = response.choices[0].message.content.strip().replace("```", "")
    if not generated_code:
        print("prompt completion with o1-mini failed, retrying with gpt-4o-mini")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": content
            }],
            max_completion_tokens=2500
        )
        generated_code = response.choices[0].message.content.strip().replace("```", "")
        print(f"gpt-4o-mini: {generated_code}")
        if not generated_code:
            raise Exception("The generated code is empty. You probably sent a too large prompt.")
    else:
        print(f"o1-mini: {generated_code}")
    
    return re.sub(r"^python\s*", "", generated_code)

@app.route('/api/validate', methods=['POST'])
def validate_student_code():
    # Validate authorization
    auth_header = request.headers.get("Authorization")
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

        if email in AUTHORIZED_USERS:
            session["user_email"] = email
            print(f'Welcome, {email}!')
        else:
            return jsonify({"error": "Unauthorized user"}), 403

    except Exception as e:
        print(f"Token validation error: {e}")
        return jsonify({"error": "Invalid token"}), 403

    # Validate request data
    data = request.get_json()
    if not data or "prompt" not in data or "function_id" not in data or "user_email" not in data:
        return jsonify({"error": "Invalid request format"}), 400

    user_prompt = data["prompt"]
    function_id = data["function_id"]
    user_email = data["user_email"]

    try:
        # Generate code from prompt
        generated_code = prompt_completion(user_prompt)
        print(f"generated code: {generated_code}")

        # Analyze code safety
        is_safe, error_message = analyze_code_safety(generated_code)
        if not is_safe:
            return jsonify({"error": f"Unsafe code: {error_message}"}), 400

        # Prepare cloud function call
        cloud_function_url = os.environ.get("GCP_FUNC_URL")
        service_account_info = json.loads(os.getenv("SERVICE_ACCOUNT_JSON"))

        credentials = service_account.IDTokenCredentials.from_service_account_info(
            service_account_info,
            target_audience=cloud_function_url
        )

        request_adapter = Request()
        credentials.refresh(request_adapter)
        
        # Call cloud function for validation
        headers = {
            'Authorization': f'Bearer {credentials.token}',
            'Content-Type': 'application/json'
        }
        
        cloud_response = requests.post(
            cloud_function_url,
            headers=headers,
            json={
                "code": generated_code,
                "function_id": function_id
            }
        )

        if cloud_response.status_code != 200:
            return jsonify({"error": f"Cloud function error: {cloud_response.text}"}), cloud_response.status_code

        # Return cloud function response
        result = cloud_response.json()
        result.update({
            "user_email": user_email,
            "function_id": function_id
        })
        
        return jsonify(result)

    except Exception as e:
        return jsonify({
            "error": f"Error processing request: {str(e)}",
            "user_email": user_email,
            "function_id": function_id
        }), 500