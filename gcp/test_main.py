def call_python_mockup():
    """
    Emulate a request to call_python for testing DataFrame validation.
    """
    from flask import Flask, request
    from main import call_python
    import pandas as pd

    app = Flask(__name__)

    # Test function that returns a DataFrame
    test_code = """
def process_dates(date_dict):
    # Criar um dataframe a partir do dicionário
    df = pd.DataFrame(date_dict)
    
    # Converter as colunas de data para o formato datetime
    df['Start Date'] = pd.to_datetime(df['Start Date'])
    df['End Date'] = pd.to_datetime(df['End Date'])
    
    # Calcular a diferença em dias entre as colunas 'End Date' e 'Start Date'
    df['Difference'] = (df['End Date'] - df['Start Date']).dt.days
    
    # Filtrar as linhas onde a diferença é maior que 10 dias
    result_df = df[df['Difference'] > 10]
    
    # Aplicar o método reset_index
    result_df.reset_index(drop=False, inplace=True)
    
    return result_df
"""

    payload = {
        "code": test_code,
        "function_id": "A8-E7"
    }

    with app.test_request_context(
            path='/mock_path',
            method='POST',
            json=payload
    ):
        response = call_python(request)
        print("Emulated Response:", response[0].get_json())


if __name__ == "__main__":
    call_python_mockup()