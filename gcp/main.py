import pandas as pd
from flask import jsonify


def call_python(request): #, DEBUG=False
    # if DEBUG:
    #     data = request
    # else:
    #     data = request.get_json()

    data = request.get_json()
    code = data["code"]


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
        result = func(*inputs)
    else:
        result = func()

    return jsonify({"status": "success", "output": result}), 200

