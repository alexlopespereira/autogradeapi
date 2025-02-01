from openai import OpenAI
import os

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

def test_openai():
    try:
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")  # Load from environment variable
        )

        response = client.chat.completions.create(
            model="o3-mini-2025-01-31",  # or "gpt-4" depending on your needs
            messages=[
                {"role": "user", "content": "Hello"},
            ],
            temperature=0.7,
            max_tokens=150,
            stream=False
        )

        print(response.choices[0].message.content)
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    #test_deepseek()
    test_openai() 