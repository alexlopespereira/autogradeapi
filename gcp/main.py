from flask import jsonify

def call_python(request):
    data = request.get_json()
    code = f"""import pandas as pd
import numpy as np
import requests
from collections import defaultdict
{data["code"]}"""
    print(code)
    inputs = data.get("inputs", None)
    exec_globals = {}
    exec(code, exec_globals)
    func_name = [name for name in exec_globals if callable(exec_globals[name])][-1]
    func = exec_globals[func_name]
    if inputs:
        try:
            result = func(*inputs)  # Attempt to execute the function with inputs
            print(f"Printing result: {result}")
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
        print(f"No args received. Executing function without args.")
        result = func()
        return jsonify({"status": "success", "output": result}), 200

if __name__ == '__main__':
    test_case = {"input": 5, "expected": {"1": 1, "2": 4, "3": 9, "4": 16, "5": 25}, "function_id": "A2-E1", "testcase_id": "1"}
    result = call_python({"code": "def gerar_dicionario_quadrados(n):\\n    return {str(i): i**2 for i in range(1, n + 1)}".replace("\\n", "\n"), "inputs": 5}, DEBUG=True)
    print(result)