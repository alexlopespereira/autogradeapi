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
def process_population(path):
    file_path = 'https://github.com/alexlopespereira/mba_enap/raw/main/data/originais/populacao/POP2024_20241101.xls'
    df = pd.read_excel(file_path, sheet_name="MUNICÍPIOS", skiprows=1, converters={'COD. UF': str, 'COD. MUNIC': str})
    df = df.rename(columns={df.columns[-2]: "POPULACAO",
                            df.columns[-3]: "MUNICIPIO"})
    df = df.iloc[:, :-1]  # Remove a última coluna
    df['cod_ibge7'] = df['COD. UF'] + df['COD. MUNIC']
    df = df[:-40]
    return df
    """

    payload = {
        "code": test_code,
        "function_id": "A6-E3"
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