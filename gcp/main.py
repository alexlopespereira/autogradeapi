import pandas as pd
from flask import jsonify
from markupsafe import escape


def pandas_http(request):
    try:
        data = request.get_json()

        if not data or "code" not in data or "inputs" not in data:
            return jsonify({"error": "Invalid input format. 'code' and 'inputs' are required."}), 400

        user_code = data["code"]
        inputs = data["inputs"]
        email = data["email"]

        # Execute the code
        exec_globals = {}
        exec(user_code, exec_globals)

        # Identify the function name
        func_name = [name for name in exec_globals if callable(exec_globals[name])][-1]
        func = exec_globals[func_name]

        # Call the function with inputs
        result = func(*inputs)

        return jsonify({"status": "success", "output": result}), 200

    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500