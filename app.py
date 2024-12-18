from flask import Flask, request, jsonify, render_template
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from util import test_student_code, execute_code_with_timeout, analyze_code_safety, test_cases

app = Flask(__name__)

# Initialize Flask-Limiter with a key function to use the remote IP address
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["1 per minute"]  # Global limit: 10 requests per minute
)


@app.route('/api/validate', methods=['POST'])
@limiter.limit("10 per minute")  # Specific route limit: 5 requests per minute
def validate_student_code():
    """
    Flask API endpoint to validate a student's function implementation.

    Expects a JSON payload with:
    - implementation (str): The function implementation as a string.
    - id (str): The ID of the test cases to validate against.

    Returns:
        JSON response with validation results.
    """
    try:
        # Parse the JSON payload
        data = request.get_json()

        if not data or "implementation" not in data or "id" not in data:
            return jsonify({"error": "Invalid request format. 'implementation' and 'id' are required."}), 400

        implementation_text = data["implementation"]
        test_id = data["id"]

        # Analyze code safety
        is_safe, error_message = analyze_code_safety(implementation_text)
        if not is_safe:
            return jsonify({"error": f"Unsafe code: {error_message}"}), 400

        # Filter test cases by ID
        filtered_test_cases = [tc for tc in test_cases if tc["id"] == test_id]
        if not filtered_test_cases:
            return jsonify({"error": f"No test cases found for ID: {test_id}"}), 404

        # Execute the test cases
        results = {"id": test_id, "test_results": []}
        for i, case in enumerate(filtered_test_cases):
            exec_result = execute_code_with_timeout(implementation_text, case["input"])
            if exec_result["status"] == "success":
                results["test_results"].append({
                    "test_case": i + 1,
                    "status": "passed" if exec_result["output"] == case["expected"] else "failed",
                    "output": exec_result["output"],
                    "expected": case["expected"],
                })
            else:
                results["test_results"].append({
                    "test_case": i + 1,
                    "status": exec_result["status"],
                    "error": exec_result.get("error", "")
                })

        results["all_tests_passed"] = all(
            r["status"] == "passed" for r in results["test_results"]
        )
        return jsonify(results)

    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {e}"}), 500


#
# @app.route('/api/unlimited', methods=['GET'])
# def unlimited_resource():
#     return jsonify({"message": "This endpoint has no rate limit."})

# Error handler for rate limit exceeded
@app.errorhandler(429)
def rate_limit_exceeded(e):
    return jsonify({"error": "rate limit exceeded", "message": str(e.description)}), 429


@app.route("/")
def hello_world():
    return render_template("index.html")


if __name__ == '__main__':
    app.run(debug=True)
