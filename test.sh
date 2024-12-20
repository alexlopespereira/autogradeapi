curl -X POST http://127.0.0.1:5000/api/validate \
-H "Content-Type: application/json" \
-d '{
    "implementation": "def test_function():\\n    return 2",
    "function_id": "AAA",
    "testcase_id": "BBB",
    "user_email": "user@example.com"
}'
