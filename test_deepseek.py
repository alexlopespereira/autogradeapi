from openai import OpenAI

def test_deepseek():
    try:
        client = OpenAI(
            api_key="<DeepSeek API Key>",  # Replace with your actual API key
            base_url="https://api.deepseek.com/v1"  # Added /v1 to match DeepSeek's API endpoint
        )

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "Hello"},
            ],
            stream=False
        )

        print(response.choices[0].message.content)
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    test_deepseek() 