import json
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import math

def round_down(n, decimals=1):
    multiplier = 10 ** decimals
    return math.floor(n * multiplier) / multiplier

def create_test_case_specification(
        groundtruth_df: pd.DataFrame,
        input_value: Any,
        function_id: str,
        testcase_id: str,
        sample_rows: List[Dict[str, Dict[str, Any]]],
        round_decimals: int = 1,
        row_match_threshold: float = 0.01
) -> Dict[str, Any]:
    """
    Create a test case specification from a groundtruth DataFrame.

    Args:
        groundtruth_df: The reference DataFrame containing the expected output
        input_value: The input value(s) for the test case
        function_id: The function identifier
        testcase_id: The test case identifier
        sample_rows: List of dictionaries containing filter and expected value pairs
                    Each dict should have format:
                    {
                        "filter": {"column_name": "value"},
                        "expected_values": {"column_name": "value"}
                    }
        round_decimals: Number of decimal places for rounding
        row_match_threshold: Threshold for numeric comparisons

    Returns:
        Dict containing the complete test case specification
    """
    # Get column information
    columns = list(groundtruth_df.columns)
    dtypes = {col: str(groundtruth_df[col].dtype) for col in columns}

    # Calculate aggregate statistics for numeric columns
    numeric_cols = groundtruth_df.select_dtypes(include=[np.number]).columns
    aggregation_checks = {
        "total_rows": {
            "min": len(groundtruth_df),
            "max": len(groundtruth_df)
        },
        "sum": {
            col: round_down(float(groundtruth_df[col].sum()), round_decimals)
            for col in numeric_cols
        }
    }

    # Create test case specification
    test_case = {
        "input": input_value if isinstance(input_value, list) else [input_value],
        "expected": {
            "type": "dataframe",
            "data": {
                "columns": columns,
                "sample_rows": sample_rows,
                "dtypes": dtypes,
                "aggregation_checks": aggregation_checks
            },
            "validation_rules": {
                "round_decimals": round_decimals,
                "row_match_threshold": row_match_threshold
            }
        },
        "function_id": function_id,
        "testcase_id": testcase_id
    }

    return test_case


def create_grupo():
    # Create sample DataFrame
    import random
    nomes = ['Lisa Watson', 'Gilbert Ravenell', 'Olga Crocker', 'Julie Lewis', 'Kimberly Ransome', 'Jennifer Daniels', 'Jillian Brennan', 'Mary Phinney', 'Michelle Humphreys']
    seed = 42
    N = 4
    if seed is not None:
        random.seed(seed)
    random.shuffle(nomes)

    n_grupos = len(nomes) // N
    resto = len(nomes) % N

    grupos = []
    inicio = 0
    for i in range(n_grupos):
        fim = inicio + N
        grupos.append(nomes[inicio:fim])
        inicio = fim

    if resto > 0:
        grupos.append(nomes[inicio:])

    df_data = []
    for i, grupo in enumerate(grupos):
        for nome in grupo:
            df_data.append(["Grupo " + str(i), nome])

    df = pd.DataFrame(df_data, columns=["Grupo", "Nome"])

    # Define sample conditions
    sample_rows = [
        {
            # First row to check
            "filter": {
                'Grupo': 'Grupo 2'  # How to find this row
            },
            "expected_values": { 'Nome': 'Olga Crocker' }
        }
    ]

    # Create test case
    test_case = create_test_case_specification(
        groundtruth_df=df,
        input_value=[nomes, 42, 4],
        function_id="A6-E2",
        testcase_id="1",
        sample_rows=sample_rows
    )

    return test_case



# Example usage function
def create_populacao():
    # Create sample DataFrame
    file_path = 'https://github.com/alexlopespereira/mba_enap/raw/main/data/originais/populacao/POP2024_20241101.xls'
    df = pd.read_excel(file_path, sheet_name="MUNICÍPIOS", skiprows=1, converters={'COD. UF': str, 'COD. MUNIC': str})
    df = df.rename(columns={df.columns[-2]: "POPULACAO",
                            df.columns[-3]: "MUNICIPIO"})
    df = df.iloc[:, :-1]  # Remove a última coluna
    df['cod_ibge7'] = df['COD. UF'] + df['COD. MUNIC']
    df = df[:-40]


    # Define sample conditions
    sample_rows = [
        {
            # First row to check
            "filter": {
                "UF": "DF"  # How to find this row
            },
            "expected_values": {
                "POPULACAO": 2982818
            }
        },
        {
            # First row to check
            "filter": {
                "cod_ibge7": "1100015"  # How to find this row
            },
            "expected_values": {
                "POPULACAO": 22853
            }
        }
    ]

    # Create test case
    test_case = create_test_case_specification(
        groundtruth_df=df,
        input_value=file_path,
        function_id="A6-E3",
        testcase_id="1",
        sample_rows=sample_rows
    )

    return test_case


def create_gini_6_4():
    # Create sample DataFrame
    data = {"Municipio": {
            "0": "110001 Alta Floresta D'Oeste",
            "1": "110037 Alto Alegre dos Parecis",
            "2": "110040 Alto Paraíso",
            "3": "110034 Alvorada D'Oeste",
            "4": "110002 Ariquemes"},
            "1991": {"0": 0.5983, "1": "", "2": "", "3": 0.569, "4": 0.5827},
            "2000": {"0": 0.5868, "1": 0.508, "2": 0.6256, "3": 0.6534, "4": 0.5927},
            "2010": {"0": 0.5893, "1": 0.5491, "2": 0.5417, "3": 0.5355, "4": 0.5496}}

    df = pd.DataFrame(data)
    unpivoted_df = pd.melt(df, id_vars=["Municipio"], var_name="data", value_name="gini")
    unpivoted_df["gini"] = pd.to_numeric(unpivoted_df["gini"], errors='coerce')

    # Define sample conditions
    sample_rows = [
        {
            # First row to check
            "filter": {
                "Municipio": "110001 Alta Floresta D'Oeste", 'data': '1991'  # How to find this row
            },
            "expected_values": {
                "gini": 0.5983
            }
        },
        {
            # First row to check
            "filter": {
                "Municipio": "110002 Ariquemes", 'data': '2010'  # How to find this row
            },
            "expected_values": {
                "gini": 0.5496
            }
        }
    ]

    # Create test case
    test_case = create_test_case_specification(
        groundtruth_df=unpivoted_df,
        input_value=data,
        function_id="A6-E4",
        testcase_id="1",
        sample_rows=sample_rows,
        row_match_threshold=0.001
    )

    return test_case

def codibge_6_5():
    # Create sample DataFrame
    data = {
        'Municipio': [
            "110001 Alta Floresta D'Oeste",
            "110037 Alto Alegre dos Parecis",
            "110040 Alto Paraíso",
            "110034 Alvorada D'Oeste",
            "110002 Ariquemes"
        ],
        '1991': [0.5983, None, None, 0.569, 0.5827],
        '2000': [0.5868, 0.508, 0.6256, 0.6534, 0.5927],
        '2010': [0.5893, 0.5491, 0.5417, 0.5355, 0.5496]
    }

    df = pd.DataFrame(data)
    df['cod_ibge6'] = df['Municipio'].str.split().str[0]

    # Define sample conditions
    sample_rows = [
        {
            # First row to check
            "filter": {
                "Municipio": "110001 Alta Floresta D'Oeste"
            },
            "expected_values": {
                "cod_ibge6": "110001"
            }
        },
        {
            # First row to check
            "filter": {
                "Municipio": "110002 Ariquemes",
            },
            "expected_values": {
                "cod_ibge6": "110002"
            }
        }
    ]

    # Create test case
    test_case = create_test_case_specification(
        groundtruth_df=df,
        input_value=data,
        function_id="A6-E5",
        testcase_id="1",
        sample_rows=sample_rows,
        row_match_threshold=0.001
    )

    return test_case


def morbidade_6_6():

    path: str = 'https://github.com/alexlopespereira/mba_enap/raw/refs/heads/main/data/originais/morbidade/consolidado/morbidade.csv'

    df = pd.read_csv(path, sep=';', na_values=['-', '...'], decimal=',', dtype={'cod_ibge6': str}, index_col=0)


    # Define sample conditions
    sample_rows = [
        {
            # First row to check
            "filter": {
                "cod_ibge6": "530010", "mes_ano": "2020-01-01"
            },
            "expected_values": {
                "Internações": 15827.0
            }
        },
        {
            # First row to check
            "filter": {
                "cod_ibge6": "110001", "mes_ano": "2019-12-01"
            },
            "expected_values": {
                "Internações": 193.0
            }
        }
    ]

    # Create test case
    test_case = create_test_case_specification(
        groundtruth_df=df,
        input_value=path,
        function_id="A6-E6",
        testcase_id="1",
        sample_rows=sample_rows,
        row_match_threshold=0.01
    )

    return test_case


def pibmunicipios_6_7():
    caminho_arquivo: str = 'https://github.com/alexlopespereira/mba_enap/raw/main/data/originais/pib/pib_municipios.xlsx'

    df = pd.read_excel(caminho_arquivo, dtype={'CodIBGE': str})
    df = df.iloc[:-1]

    # Define sample conditions
    sample_rows = [
        {
            # First row to check
            "filter": {
                "CodIBGE": "1100015",
            },
            "expected_values": {
                "2007": 191364
            }
        },
        {
            # First row to check
            "filter": {
                "CodIBGE": "5300108"
            },
            "expected_values": {
                "2017": 244682756
            }
        }
    ]

    # Create test case
    test_case = create_test_case_specification(
        groundtruth_df=df,
        input_value=caminho_arquivo,
        function_id="A6-E7",
        testcase_id="1",
        sample_rows=sample_rows,
        row_match_threshold=0.01
    )

    return test_case


def morbidade_desagregado_6_8():
    import requests
     # Raise an exception for bad status codes
    from io import BytesIO, StringIO
    from zipfile import ZipFile
    import re


    urls = [
            "https://github.com/alexlopespereira/mba_enap/raw/refs/heads/main/data/originais/morbidade/desagregado/2020/A103238189_28_143_208.csv",
            "https://github.com/alexlopespereira/mba_enap/raw/refs/heads/main/data/originais/morbidade/desagregado/2020/A212148189_28_143_208.csv",
            "https://github.com/alexlopespereira/mba_enap/raw/refs/heads/main/data/originais/morbidade/desagregado/2020/A212323189_28_143_208.csv",
            "https://github.com/alexlopespereira/mba_enap/raw/refs/heads/main/data/originais/morbidade/desagregado/2020/A212334189_28_143_208.csv",
            "https://github.com/alexlopespereira/mba_enap/raw/refs/heads/main/data/originais/morbidade/desagregado/2020/A212345189_28_143_208.csv",
            "https://github.com/alexlopespereira/mba_enap/raw/refs/heads/main/data/originais/morbidade/desagregado/2021/A152003189_28_143_208.csv",
            "https://github.com/alexlopespereira/mba_enap/raw/refs/heads/main/data/originais/morbidade/desagregado/2021/A152053189_28_143_208.csv",
            "https://github.com/alexlopespereira/mba_enap/raw/refs/heads/main/data/originais/morbidade/desagregado/2021/A152118189_28_143_208.csv",
            "https://github.com/alexlopespereira/mba_enap/raw/refs/heads/main/data/originais/morbidade/desagregado/2021/A152142189_28_143_208.csv"
            ]
    
   
    dados_concatenados = pd.DataFrame()

    for url in urls:
        resposta = requests.get(url)
        conteudo = resposta.content.decode('iso-8859-1')

        linhas = conteudo.splitlines()
        linha_periodo = linhas[2].strip()

        if "Período:" in linha_periodo:
            # Tratamento do padrão esperado
            periodo = linha_periodo.split("Período:")[1].strip().rstrip(';')
            mes, ano = periodo.split('/')
            mes_numero = {"Jan": 1, "Fev": 2, "Mar": 3, "Abr": 4, "Mai": 5, "Jun": 6,
                            "Jul": 7, "Ago": 8, "Set": 9, "Out": 10, "Nov": 11, "Dez": 12}
            mes = mes_numero[mes]
            ano = int(ano)
        else:
            # Tratamento do padrão diferente
            periodo = linha_periodo.split("Período:")[1].strip().rstrip(';')
            mes, ano = periodo.split('/')
            mes_numero = 4  # Para 'Abr'
            ano = int(ano)

        # Carregando o arquivo para um DataFrame
        df = pd.read_csv(StringIO(resposta.text), encoding='iso8859-1', skiprows=3, sep=';', skipfooter=7)

        # Adicionando colunas de mês e ano
        df['month'] = mes
        df['year'] = ano

        # Criando a coluna 'Data' com o primeiro dia do mês
        df['Data'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str) + '-01')

        # Concatenando os dados
        dados_concatenados = pd.concat([dados_concatenados, df], ignore_index=True)

    sample_rows = [
        {
            # First row to check
            "filter": {
                "Município": "110001 Alta Floresta D'Oeste", "Data": "2020-09-01"
            },
            "expected_values": {
                "Internações": 146
            }
        },
        {
            # First row to check
            "filter": {
                "Município": "530010 Brasília", "Data": "2021-01-01"
            },
            "expected_values": {
                "Internações": 19145
            }
        }
    ]

    # Create test case
    test_case = create_test_case_specification(
        groundtruth_df=dados_concatenados,
        input_value=[urls],
        function_id="A6-E8",
        testcase_id="1",
        sample_rows=sample_rows,
        row_match_threshold=0.9
    )

    return test_case


def merge_pib_pop_7_1_a_7_3():
    pop_url = 'https://github.com/alexlopespereira/mba_enap/raw/refs/heads/main/data/originais/populacao/estimativa_dou_2017.xlsx'
    pib_url = 'https://github.com/alexlopespereira/mba_enap/raw/main/data/originais/pib/pib_municipios.xlsx'

    df_pop = pd.read_excel(pop_url, dtype={'cod_ibge': str})
    df_pib = pd.read_excel(pib_url, skipfooter=1, dtype={'CodIBGE': str})
    df_merged = pd.merge(df_pop, df_pib, left_on='cod_ibge', right_on='CodIBGE')
    df_merged['pib_percapita'] = df_merged['2017'] / df_merged['pop2017']
    df_merged['pib_percapita'] = df_merged['pib_percapita'].apply(lambda x: float(f'{x:.1f}'))
    top_10_pib_percapita = df_merged.nlargest(10, 'pib_percapita')
    sample_rows = [
        {
            # First row to check
            "filter": {
                "cod_ibge": "3536505"
            },
            "expected_values": {
                "pib_percapita": 344.8
            }
        },
        {
            # First row to check
            "filter": {
                "cod_ibge": "3524709"
            },
            "expected_values": {
                "pib_percapita": 209.3
            }
        }
    ]

    # Create test case
    test_case = create_test_case_specification(
        groundtruth_df=top_10_pib_percapita,
        input_value=[pop_url, pib_url],
        function_id="A7-E3",
        testcase_id="1",
        sample_rows=sample_rows,
        row_match_threshold=0.01
    )

    return test_case


def merge_morbidade_pop_7_4_7_5():
    path_morb = 'https://github.com/alexlopespereira/mba_enap/raw/refs/heads/main/data/originais/morbidade/consolidado/morbidade.csv'
    url_pop = 'https://github.com/alexlopespereira/mba_enap/raw/refs/heads/main/data/originais/populacao/estimativa_dou_2017.xlsx'

    df_morbidade = pd.read_csv(path_morb, sep=';', na_values=['-', '...'], decimal=',', index_col=0, dtype={'cod_ibge6': str})

    # Load the Excel file from the URL
    df_pop = pd.read_excel(url_pop, dtype={'cod_ibge': str}, engine='openpyxl')

    # Create the 'cod_ibge6' column in df_pop
    df_pop['cod_ibge6'] = df_pop['cod_ibge'].str[:-1]

    # Merge the two DataFrames
    df_merged = pd.merge(df_morbidade, df_pop, on='cod_ibge6', how='left')

    # Calculate morbidade_pormil
    df_merged['morbidade_pormil'] = (df_merged['Óbitos'] / df_merged['pop2017']) * 1000
    df_merged['morbidade_pormil'] = df_merged['morbidade_pormil'].round(1)

    # Filter for the year 2020 and sort by morbidade_pormil in descending order
    df_2020 = df_merged[(df_merged['ano'] == 2020) & (df_merged['mes'] == 'Dez')].sort_values(by='morbidade_pormil', ascending=False)

    # Return the top 10 records with the highest morbidade_pormil for the year 2020
    df10 = df_2020.head(10)

    sample_rows = [
        {
            # First row to check
            "filter": {
                "cod_ibge6": "350720", "mes_ano": "2020-12-01"
            },
            "expected_values": {
                "morbidade_pormil": 3.6
            }
        },
        {
            # First row to check
            "filter": {
                "cod_ibge6": "316830", "mes_ano": "2020-12-01"
            },
            "expected_values": {
                "morbidade_pormil": 1.5
            }
        }
    ]

    # Create test case
    test_case = create_test_case_specification(
        groundtruth_df=df10,
        input_value=[path_morb, url_pop],
        function_id="A7-E5",
        testcase_id="1",
        sample_rows=sample_rows,
        row_match_threshold=0.01
    )

    return test_case


def pop_pib_gini_7_6_e_7_7():
    path_pib = 'https://github.com/alexlopespereira/mba_enap/raw/main/data/originais/pib/pib_municipios.xlsx'
    path_pop = 'https://github.com/alexlopespereira/mba_enap/raw/refs/heads/main/data/originais/populacao/estimativa_dou_2017.xlsx'
    path_gini = 'https://github.com/alexlopespereira/mba_enap/raw/refs/heads/main/data/originais/gini/ginibr.csv'

    df_pib = pd.read_excel(path_pib, skipfooter=1, dtype={'CodIBGE': str}, engine='openpyxl')
    df_pib['cod_ibge6'] = df_pib['CodIBGE'].str[:-1]
    df_pib[['2007', '2009', '2011']] = df_pib[['2007', '2009', '2011']].astype(str)
    df_pib[['2013', '2015', '2017']] = df_pib[['2013', '2015', '2017']].astype('float64')

    # Load population data
    df_pop = pd.read_excel(path_pop, dtype={'cod_ibge': str}, engine='openpyxl')

    # Merge PIB and population data
    df_merge = pd.merge(df_pop, df_pib, left_on='cod_ibge', right_on='CodIBGE', how='inner')

    # Calculate PIB per capita
    df_merge['pib_percapita'] = (df_merge['2017'] / df_merge['pop2017']).round(1)

    # Load Gini index data
    df_gini = pd.read_csv(path_gini, sep=';', na_values='...', decimal=',', skiprows=2, skipfooter=2, encoding='utf8', engine='python')
    df_gini['cod_ibge6'] = df_gini['Município'].str.split(' ').str[0]

    # Merge with Gini index data
    df_final = pd.merge(df_merge, df_gini, on='cod_ibge6', how='inner')

    df_top_pib = df_final.sort_values(by=['pib_percapita'], ascending=False).head(10)
    df_top_gini = df_final.sort_values(by=['2010'], ascending=False).head(10)
    df20 = pd.concat([df_top_pib, df_top_gini])

    sample_rows = [
        {
            # First row to check
            "filter": {
                "cod_ibge6": "353650"
            },
            "expected_values": {
                "2010": 0.4880
            }
        },
        {
            # First row to check
            "filter": {
                "cod_ibge6": "140002"
            },
            "expected_values": {
                "2010": 0.7502
            }
        }
    ]

    # Create test case
    test_case = create_test_case_specification(
        groundtruth_df=df20,
        input_value=[path_pib, path_pop, path_gini],
        function_id="A7-E7",
        testcase_id="1",
        sample_rows=sample_rows,
        row_match_threshold=0.01
    )
    return test_case


def dias_uteis_8_6():
    import datetime
    from datetime import date, timedelta
    ano = 2024
    ultimos_dias_uteis = []

    s = pd.date_range(f'{ano}-01-01', periods=12, freq='BM')
    df = pd.DataFrame(s, columns=['Col1'])
    df['Col2'] = df['Col1'].dt.day_name()

    sample_rows = [
        {
            # First row to check
            "filter": {
                "Col1": "2024-01-31"
            },
            "expected_values": {
                "Col2": "Wednesday"
            }
        },
        {
            "filter": {
                "Col1": "2024-12-31"
            },
            "expected_values": {
                "Col2": "Tuesday"
            }
        }
    ]

    # Create test case
    test_case = create_test_case_specification(
        groundtruth_df=df,
        input_value=2024,
        function_id="A8-E1",
        testcase_id="1",
        sample_rows=sample_rows,
        row_match_threshold=0.01
    )
    return test_case


def date_diff_87():
    data = {
    'Start Date': ['2021-05-01', '2021-06-15', '2021-07-20'],
    'End Date': ['2021-05-10', '2021-06-25', '2021-08-01']
    }
    df = pd.DataFrame(data)

    # Convert date columns to datetime
    df['Start Date'] = pd.to_datetime(df['Start Date'])
    df['End Date'] = pd.to_datetime(df['End Date'])

    # Calculate the difference in days
    df['Difference'] = (df['End Date'] - df['Start Date']).dt.days

    # Filter and return results
    df2 = df[df['Difference'] > 10]

    sample_rows = [
        {
            # First row to check
            "filter": {
                "index": 2
            },
            "expected_values": {
                "Difference": 2
            }
        }
    ]

    # Create test case
    test_case = create_test_case_specification(
        groundtruth_df=df2,
        input_value=data,
        function_id="A8-E7",
        testcase_id="1",
        sample_rows=sample_rows,
        row_match_threshold=0.01
    )
    return test_case


def full_week_diff_8_8():
    data = {
        'Event': ['Event A', 'Event B', 'Event C'],
        'Start Date': ['2023-01-05', '2023-02-15', '2023-07-20'],
        'End Date': ['2023-02-10', '2023-03-28', '2023-08-01']
    }
    df = pd.DataFrame(data)
    df['Start Date'] = pd.to_datetime(df['Start Date'])
    df['End Date'] = pd.to_datetime(df['End Date'])
    df['Difference'] = (df['End Date'] - df['Start Date']).dt.days
    df['Full Weeks Difference'] = df['Difference'] // 7

    sample_rows = [
        {
            # First row to check
            "filter": {
                "Event": "Event A"
            },
            "expected_values": {
                "Full Weeks Difference": 5
            }
        }
    ]

    # Create test case
    test_case = create_test_case_specification(
        groundtruth_df=df,
        input_value=data,
        function_id="A8-E8",
        testcase_id="1",
        sample_rows=sample_rows,
        row_match_threshold=0.01
    )
    return test_case

def transformar_dicionario_em_dataframe_8_4():
    data = {"index": ["2011-09-02", "2012-08-04", "2013-09-03"], "0": [-0.96, -0.05, -0.42]}
    df = pd.DataFrame(data)

    # Adicionar a coluna 'mes' com o número do mês
    df['mes'] = pd.to_datetime(df['index']).dt.month

    sample_rows = [
        {
            # First row to check
            "filter": {
                "0": -0.96
            },
            "expected_values": {
                "mes": 9
            }
        }
    ]

    # Create test case
    test_case = create_test_case_specification(
        groundtruth_df=df,
        input_value=data,
        function_id="A8-E4",
        testcase_id="1",
        sample_rows=sample_rows,
        row_match_threshold=0.01
    )
    return test_case


def aggregate_population_data_9_1():
    url = 'https://github.com/alexlopespereira/mba_enap/raw/refs/heads/main/data/originais/populacao/estimativa_dou_2017.xlsx'
    try:
        df_pop = pd.read_excel(url, dtype={'cod_ibge': str})
        df_pop_grouped = df_pop.groupby('uf')['pop2017'].agg(['sum', 'mean', 'median', 'std'])
        df_pop_grouped = df_pop_grouped.round(1)
        df_pop_grouped = df_pop_grouped.sort_values(by='sum', ascending=False)
        df_pop_grouped = df_pop_grouped.head(3).reset_index(drop=False)
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

    sample_rows = [
        {
            # First row to check
            "filter": {
                "uf": "SP"
            },
            "expected_values": {
                "sum": 45094866
            }
        },
        {
            # First row to check
            "filter": {
                "uf": "RJ"
            },
            "expected_values": {
                "sum": 16718956
            }
        }
    ]

    # Create test case
    test_case = create_test_case_specification(
        groundtruth_df=df_pop_grouped,
        input_value=url,
        function_id="A9-E1",
        testcase_id="1",
        sample_rows=sample_rows,
        row_match_threshold=0.01
    )
    return test_case


def aggregate_titanic_data_9_2():
    url = "https://github.com/alexlopespereira/mba_enap/raw/main/data/originais/titanic/titanic.xls"
    try:
        df_titanic = pd.read_excel(url)
        df_agg = df_titanic.groupby(['pclass', 'sex']).agg({'survived': 'sum', 'fare': 'median'})
        df_agg = df_agg.round(1)
        df_agg = df_agg.reset_index(drop=False)
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

    sample_rows = [
        {
            # First row to check
            "filter": {
                "pclass": 1, "sex": "female"
            },
            "expected_values": {
                "survived": 139
            }
        },
        {
            # First row to check
            "filter": {
                "pclass": 3, "sex": "male"
            },
            "expected_values": {
                "survived": 75
            }
        }
    ]

    # Create test case
    test_case = create_test_case_specification(
        groundtruth_df=df_agg,
        input_value=url,
        function_id="A9-E2",
        testcase_id="1",
        sample_rows=sample_rows,
        row_match_threshold=0.01
    )
    return test_case


def calcula_custo_medio_internacao_9_3():
    url = 'https://github.com/alexlopespereira/mba_enap/raw/refs/heads/main/data/originais/morbidade/consolidado/morbidade.csv'
    df_morbidade = pd.read_csv(url, sep=';', na_values=['-', '...'], decimal=',', dtype={'cod_ibge6': str}, index_col=0)
    df_morbidade['uf'] = df_morbidade['cod_ibge6'].str[:2]
    df_morbidade = df_morbidade.groupby(['uf', 'cod_ibge6', 'Município'])[['Valor_total', 'Internações', 'Dias_permanência']].sum().round(1)
    df_morbidade['custo_medio_diario_intern'] = (df_morbidade['Valor_total'] / df_morbidade['Dias_permanência']).round(1)
    df_morbidade = df_morbidade.reset_index(drop=False)
    df_morbidade = df_morbidade.sort_values('custo_medio_diario_intern', ascending=False).head(5)


    sample_rows = [
        {
            # First row to check
            "filter": {
                "cod_ibge6": "410400"
            },
            "expected_values": {
                "custo_medio_diario_intern": 1137.2
            }
        },
        {
            # First row to check
            "filter": {
                "cod_ibge6": "330115"
            },
            "expected_values": {
                "custo_medio_diario_intern": 729.7
            }
        }
    ]

    # Create test case
    test_case = create_test_case_specification(
        groundtruth_df=df_morbidade,
        input_value=url,
        function_id="A9-E3",
        testcase_id="1",
        sample_rows=sample_rows,
        row_match_threshold=0.01
    )
    return test_case


def soma_valor_total_municipios_9_4():
    """
    Calcula a soma do Valor_total para os municípios especificados.

    Args:
        url (str): URL do arquivo CSV.
        lista_cod_ibge6 (list): Lista de códigos IBGE6 dos municípios.

    Returns:
        pandas.DataFrame: DataFrame com a soma do Valor_total por município.
    """
    url = 'https://github.com/alexlopespereira/mba_enap/raw/refs/heads/main/data/originais/morbidade/consolidado/morbidade.csv'
    lista_codigos = ["410400", "330115", "330205", "353715", "355365"]
    df_morbidade = pd.read_csv(url, sep=';', na_values=['-', '...'], decimal=',', dtype={'cod_ibge6': str}, index_col=0)
    df_morbidade['uf'] = df_morbidade['cod_ibge6'].str[:2]
    df_morbidade = df_morbidade[df_morbidade['cod_ibge6'].isin(lista_codigos)]
    df_morbidade = df_morbidade.groupby('Município')['Valor_total'].sum().round(1).reset_index(drop=False)


    sample_rows = [
        {
            # First row to check
            "filter": {
                "Município": "Campina Grande do Sul 	"
            },
            "expected_values": {
                "Valor_total": 1137.2
            }
        },
        {
            # First row to check
            "filter": {
                "cod_ibge6": "330115"
            },
            "expected_values": {
                "custo_medio_diario_intern": 729.7
            }
        }
    ]

    # Create test case
    test_case = create_test_case_specification(
        groundtruth_df=df_morbidade,
        input_value=url,
        function_id="A9-E4",
        testcase_id="1",
        sample_rows=sample_rows,
        row_match_threshold=0.01
    )
    return test_case


if __name__ == "__main__":
    # Example usage
    test_case = morbidade_desagregado_6_8() #create_grupo()
    print("Generated Test Case:")
    print(json.dumps(test_case).replace("\n",""))
    




    
