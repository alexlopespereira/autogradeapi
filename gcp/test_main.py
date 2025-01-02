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
def process_morbidade_data(caminho_arquivo):
    df_morbidade = pd.read_csv(caminho_arquivo, sep=";", na_values=['-', '...'], decimal=',', dtype={'cod_ibge6': str}, index_col=0)
    
    df_morbidade['uf'] = df_morbidade['cod_ibge6'].str[:2]
    
    agg_df = df_morbidade.groupby(['uf', 'cod_ibge6', 'Município']).agg({
        'Valor_total': 'sum',
        'Internações': 'sum',
        'Dias_permanência': 'sum'
    }).reset_index(drop=False)
    
    agg_df['Valor_total'] = agg_df['Valor_total'].round().astype('int64')
    agg_df['Internações'] = agg_df['Internações'].round().astype('int64')
    agg_df['Dias_permanência'] = agg_df['Dias_permanência'].round().astype('int64')
    
    agg_df['custo_medio_diario_intern'] = agg_df['Valor_total'] / agg_df['Dias_permanência']
    
    return agg_df.nlargest(5, 'custo_medio_diario_intern')
"""

    payload = {
        "code": test_code,
        "function_id": "A9-E3"
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