import json
import os
from openai import OpenAI

# Configure sua chave de API da OpenAI
# É recomendado usar variáveis de ambiente para segurança.
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Caminhos para os arquivos de entrada e saída
input_file_path = 'c:\\Projects\\autogradeapi\\gcp_prompt\\answer_prompts.json'
output_file_path = 'c:\\Projects\\autogradeapi\\gcp_prompt\\answer_prompts_novo.json'

def process_prompts():
    """
    Lê prompts de um arquivo JSON, obtém o código correspondente da API OpenAI
    e salva os resultados em um novo arquivo JSON.
    """
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Erro: O arquivo de entrada não foi encontrado em '{input_file_path}'")
        return
    except json.JSONDecodeError:
        print(f"Erro: O arquivo de entrada em '{input_file_path}' não é um JSON válido.")
        return

    for item in data:
        if 'prompt' in item:
            try:
                print(f"Processando prompt: {item['prompt'][:50]}...")
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",  # Ou o modelo que preferir
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that provides only Python code based on the user's prompt. Do not include any markdown, formatting, or explanations."},                        
                        {"role": "user", "content": item['prompt']}
                    ]
                )
                
                code = response.choices[0].message.content.strip()
                # Remove ```python and ``` if they exist
                if code.startswith('```python'):
                    code = code[len('```python'):].strip()
                if code.endswith('```'):
                    code = code[:-len('```')].strip()

                item['code'] = code
                print("Código recebido e adicionado.")

            except Exception as e:
                print(f"Ocorreu um erro ao processar o prompt: {e}")
                item['code'] = f"Erro ao gerar código: {e}"

    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Arquivo processado e salvo com sucesso em '{output_file_path}'")
    except IOError as e:
        print(f"Ocorreu um erro ao salvar o arquivo de saída: {e}")

if __name__ == "__main__":
    # Verifique se a chave da API está configurada
    if not os.environ.get("OPENAI_API_KEY"):
        print("Erro: A variável de ambiente OPENAI_API_KEY não está configurada.")
        print("Por favor, configure a chave da API antes de executar o script.")
    else:
        process_prompts()
