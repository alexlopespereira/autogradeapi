import hashlib
import os
import pandas as pd

def hash_email(email: str, salt: str) -> str:
    """Hash email with provided salt using md5."""
    return hashlib.md5(f"{email}{salt}".encode()).hexdigest()

def main():
    # Get salt from environment variable or use a default for testing
    salt = os.environ.get("EMAIL_HASH_SALT", "teste_hash")
    
    # Read the CSV file
    df = pd.read_csv('MBA_Enap_2024_Submissions - turma.csv')
    
    # Add hash column using the 'Email novo' column
    df['Email Hash'] = df['Email novo'].apply(lambda x: hash_email(x, salt))
    
    # Save the updated CSV
    df.to_csv('MBA_Enap_2024_Submissions - turma_with_hash.csv', index=False)
    print("CSV file has been updated with hash column and saved as 'MBA_Enap_2024_Submissions - turma_with_hash.csv'")

if __name__ == "__main__":
    main() 