import unittest
from unittest.mock import patch, MagicMock
from app import app
import json

class TestValidateStudentCode(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()
        
        # Mock authorized users
        self.authorized_email = "alexlopespereira@gmail.com"
        global AUTHORIZED_USERS
        AUTHORIZED_USERS = {self.authorized_email}

    @patch('requests.get')
    @patch('requests.post')
    @patch('app.prompt_completion')
    def test_validate_student_code_success(self, mock_prompt_completion, mock_post, mock_get):
        # Mock Google token validation response
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "email": self.authorized_email
        }

        # Mock generated code
        mock_prompt_completion.return_value = "def test_function(): return 42"

        # Mock cloud function response
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "test_results": [{"passed": True}],
            "error": None
        }

        # Test data
        test_data = {
            "prompt": "Write a function that returns 42",
            "function_id": "test_function",
            "user_email": self.authorized_email
        }

        # Make request
        response = self.client.post(
            '/api/validate',
            headers={
                'Authorization': 'Bearer fake_token',
                'Content-Type': 'application/json'
            },
            data=json.dumps(test_data)
        )

        # Assert response
        self.assertEqual(response.status_code, 200)
        response_data = json.loads(response.data)
        self.assertTrue(all(test["passed"] for test in response_data["test_results"]))

if __name__ == '__main__':
    unittest.main() 