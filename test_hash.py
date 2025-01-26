import os
import re
import json
import requests

import json
import re
import requests
from datetime import datetime, timedelta
from googleapiclient.discovery import build
from openai import OpenAI
from google.cloud import secretmanager
import hashlib
import io


def parse_exercise_files(file_paths):
    exercises = []
    
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Extract class number from filename (e.g., "Aula3.txt" -> "3")
        class_num = re.search(r'Aula(\d+)\.txt', file_path).group(1)
        
        # Find all exercises in file
        exercise_matches = re.finditer(r'(\d+\.\d+)\s+(.*?)(?=\d+\.\d+|\Z)', content, re.DOTALL)
        
        for match in exercise_matches:
            exercise_num = match.group(1).split('.')[1]
            prompt = match.group(2).strip()
            
            exercise = {
                "function_id": f"A{class_num}-E{exercise_num}",
                "prompt": prompt
            }
            exercises.append(exercise)
    
    return exercises

def generate_code(prompt, api_key):
    """
    Call DeepSeek API to generate code for a given prompt
    """
    api_key = "KEY"
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com/v1"
    )


    response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[{"role": "user", "content": f"In your answer do not return in hypertext format, return only raw text. Do not produce code for importing packages, all the allowed packages are already imported. Do not create code for testing the function. Write a Python function for the following prompt:\n{prompt}"}],
            max_tokens=2500
        )
    generated_response = response.choices[0].message.content.strip()

    return generated_response
    

def main():
    # List of exercise files
    files = [
        "data/Soluções/Aula2.txt",
        "data/Soluções/Aula3.txt",
        "data/Soluções/Aula4.txt",
        "data/Soluções/Aula5.txt",
        "data/Soluções/Aula6.txt",
        "data/Soluções/Aula7.txt",
        "data/Soluções/Aula8.txt",
        "data/Soluções/Aula9.txt"
    ]
    
    # Parse exercises
    exercises = parse_exercise_files(files)
    
    # Generate code for each exercise
    api_key = os.getenv("DEEPSEEK_API_KEY")
    
    for exercise in exercises:
        code = generate_code(exercise["prompt"], api_key)
        exercise["code"] = code
    
    # Save to JSON file
    with open("data/answer_prompts.json", "w", encoding="utf-8") as f:
        json.dump(exercises, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()