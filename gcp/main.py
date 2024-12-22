import pandas as pd
from flask import jsonify


def call_python(request): #, DEBUG=False): #
    # if DEBUG:
    #     data = request
    # else:
    #     data = request.get_json()

    data = request.get_json()
    code = data["code"]
    print(code)

    inputs = data.get("inputs", None)

    # Execute the code
    exec_globals = {}
    exec(code, exec_globals)
    #
    # # Identify the function name
    func_name = [name for name in exec_globals if callable(exec_globals[name])][-1]
    func = exec_globals[func_name]
    #
    # # Call the function with inputs
    if inputs:
        try:
            result = func(*inputs)  # Attempt to execute the function with inputs
            return jsonify({"status": "success", "output": result}), 200
        except Exception as e:
            print(str(e))
            # Handle the exception and return a JSON response with an HTTP error code
            return jsonify({
                "output": "error",
                "status": "error",
                "message": "An error occurred while executing the function.",
                "error": str(e)
            }), 500
    else:
        result = func()
        return jsonify({"status": "success", "output": result}), 200

if __name__ == '__main__':
    test_case = {"input": 5, "expected": {"1": 1, "2": 4, "3": 9, "4": 16, "5": 25}, "function_id": "A2-E1", "testcase_id": "1"}
    result = call_python({"code": "def gerar_dicionario_quadrados(n):\\n    return {str(i): i**2 for i in range(1, n + 1)}".replace("\\n", "\n"), "inputs": 5}, DEBUG=True)
    print(result)