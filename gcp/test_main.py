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
def carregar_csv(urls):



    dataframes = []
    
    for url in urls:
        resposta = requests.get(url)
        conteudo = resposta.content.decode('iso-8859-1')
        
        # Extrair período da terceira linha
        linhas = conteudo.splitlines()
        terceira_linha = linhas[2]
        
        if "Período:" in terceira_linha:
            match = re.search(r'Período:([A-Za-z]{3}/\d{4})', terceira_linha)
            if match:
                periodo = match.group(1)
                mes_str, ano_str = periodo.split('/')
                ano = int(ano_str)
                
                # Converter mês de string para número
                meses = {'Jan': 1, 'Fev': 2, 'Mar': 3, 'Abr': 4, 'Mai': 5, 'Jun': 6,
                         'Jul': 7, 'Ago': 8, 'Set': 9, 'Out': 10, 'Nov': 11, 'Dez': 12}
                mes = meses.get(mes_str)
                
                # Carregar o dataframe
                df = pd.read_csv(BytesIO(resposta.content), encoding='iso-8859-1', skiprows=3, sep=';', skipfooter=7)
                
                # Adicionar colunas de mês e ano
                df['month'] = mes
                df['year'] = ano
                
                # Criar coluna de Data
                df['Data'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str) + '-01')
                
                # Adicionar DataFrame à lista
                dataframes.append(df)
    
    # Concatenar todos os dataframes
    df_final = pd.concat(dataframes, ignore_index=True)
    return df_final
"""

    payload = {
        "code": test_code,
        "function_id": "A6-E8"
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