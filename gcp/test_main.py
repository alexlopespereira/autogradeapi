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
def carregar_e_extraire_dados(url):
    # Baixar o arquivo zip da URL
    resposta = requests.get(url)
    zip_file = zipfile.ZipFile(io.BytesIO(resposta.content))

    # Inicializar uma lista para armazenar dataframes
    lista_dataframes = []

    # Iterar por cada arquivo no arquivo zip
    for nome_arquivo in zip_file.namelist():
        if nome_arquivo.endswith('.csv'):
            with zip_file.open(nome_arquivo) as file:
                # Ler o arquivo CSV, pulando as 3 primeiras linhas
                df = pd.read_csv(file, skiprows=3, sep=';', encoding='iso8859-1', skipfooter=7, engine='python')

                # Extrair o padrão específico da 3ª linha para mês e ano
                with zip_file.open(nome_arquivo) as file:
                    linhas = file.readlines()
                    if len(linhas) >= 3:
                        linha_3 = linhas[2].decode('iso8859-1').strip()
                        if linha_3.startswith("Período:"):
                            # Extraindo mês e ano
                            if ";;;;;;;;;" in linha_3:
                                mes_ano = linha_3.split(":")[1].split(";;;;;;;")[0].strip()
                            else:
                                mes_ano = linha_3.split(":")[1].strip()
                            mes, ano = mes_ano.split("/")
                            # Adicionando colunas ao dataframe
                            df['mes'] = int(mes)
                            df['ano'] = int(ano)
                            # Criar a coluna Data
                            df['Data'] = pd.to_datetime(df['ano'].astype(str) + '-' + df['mes'].astype(str) + '-01')

                            # Adicionar o dataframe à lista
                            lista_dataframes.append(df)

    # Concatenar todos os dataframes em um único
    dataframe_final = pd.concat(lista_dataframes, ignore_index=True)

    return dataframe_final
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