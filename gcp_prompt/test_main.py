def validate_code_mockup():
    """
    Emulate a request to call_python for testing DataFrame validation.
    """
    from flask import Flask, request
    from main import validate_code
    import pandas as pd

    app = Flask(__name__)

    # Test function that returns a DataFrame
    
    payload = {
        "prompt": "Write me a Python function chamada repeat que repita cada um dos numeros de uma lista a mesma quantidade de vezes do proprio numero numa nova lista e a retorne",
        "function_id": "A2-E1",
        "user_email": "alexlopespereira@gmail.com",
        "course": "mba_enap"
    }

    with app.test_request_context(
            path='/mock_path',
            method='POST',
            json=payload
    ):
        response = validate_code(request)
        print("Emulated Response:", response[0].get_json())


if __name__ == "__main__":
    validate_code_mockup()