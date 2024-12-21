curl -X POST http://127.0.0.1:5000/api/validate \
-H "Content-Type: application/json; charset=utf-8" \
-d '{
    "prompt": "crie uma função que recebe dois argumentos numericos e retorna a soma deles",
    "function_id": "A1-E2",
    "user_email": "user@example.com"
}'