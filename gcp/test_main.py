


# Emulating a request to pandas_http
def call_python_mockup():
    """
    Emulate a request to pandas_http for development/testing purposes.
    """
    from flask import Flask, request

    from main import call_python

    app = Flask(__name__)
    # Create a mock request payload
    payload = {
    "prompt": "crie uma função que recebe dois argumentos numericos e retorna a soma deles",
    "function_id": "AAA",
    "inputs": (3, 5),
    "user_email": "user@example.com"
    }

    # Using Flask's test_request_context to create a mock request
    with app.test_request_context(
        path='/mock_path',
        method='POST',
        json=payload
    ):
        # Directly call the pandas_http function
        response = call_python(request)
        print("Emulated Response:", response[0].get_json())

# Run the test
if __name__ == "__main__":
    # Run the test function to emulate the request
    call_python_mockup()