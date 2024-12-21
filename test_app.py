import requests
from requests.sessions import Session

BASE_URL = "http://127.0.0.1:5000"

def test_login_and_validate():
    # Step 1: Start a session to maintain cookies
    session = Session()

    # Step 2: Call the /login endpoint to initiate the OAuth flow
    login_response = session.get(f"{BASE_URL}/login", allow_redirects=True)
    print("Login Response URL:", login_response.url)

    # Step 3: Simulate user authorization (mock this step or use a real browser-based flow)
    # Normally, you'd handle this in the browser. For testing purposes, you may extract the redirect URL.
    # Example: assuming the flow completes and the app redirects to /callback with tokens.

    # Mock redirect to /callback with tokens
    callback_url = f"{BASE_URL}/callback?code=mock_oauth_code&state=mock_state"
    callback_response = session.get(callback_url)
    print("Callback Response JSON:", callback_response.json())

    # Step 4: Call the /api/validate endpoint after login
    validate_payload = {
        "key": "value"  # Replace with your actual payload
    }
    validate_response = session.post(f"{BASE_URL}/api/validate", json=validate_payload)
    print("Validate Response Status Code:", validate_response.status_code)
    print("Validate Response JSON:", validate_response.json())

# Run the test
if __name__ == "__main__":
    test_login_and_validate()
