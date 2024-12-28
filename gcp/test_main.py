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
def ler_arquivo_excel(caminho):
    df = pd.read_excel(caminho)
    df = df.iloc[:-1]
    df['Unnamed: 1'] = df['Unnamed: 1'].astype(str)
    df = df.rename(columns={
        'Unnamed: 0': 'nivel',
        'Unnamed: 1': 'cod_ibge7',
        'Unnamed: 2': 'municipio'
    })
    return df
"""

    payload = {
        "code": test_code,
        "function_id": "A6-E7"
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