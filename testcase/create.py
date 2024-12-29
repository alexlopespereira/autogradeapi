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
        },
        "mean": {
            col: round_down(float(groundtruth_df[col].mean()), round_decimals)
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

    urls = [
        "https://github.com/alexlopespereira/mba_enap/raw/refs/heads/main/data/originais/morbidade/desagregado/2019/A002344189_28_143_208.csv",
        "https://github.com/alexlopespereira/mba_enap/raw/refs/heads/main/data/originais/morbidade/desagregado/2019/A212356189_28_143_208.csv",
        "https://github.com/alexlopespereira/mba_enap/raw/refs/heads/main/data/originais/morbidade/desagregado/2019/A212407189_28_143_208.csv",
        "https://github.com/alexlopespereira/mba_enap/raw/refs/heads/main/data/originais/morbidade/desagregado/2020/A002126189_28_143_208.csv",
        "https://github.com/alexlopespereira/mba_enap/raw/refs/heads/main/data/originais/morbidade/desagregado/2020/A102654189_28_143_208.csv",
        "https://github.com/alexlopespereira/mba_enap/raw/refs/heads/main/data/originais/morbidade/desagregado/2020/A102744189_28_143_208.csv",
        "https://github.com/alexlopespereira/mba_enap/raw/refs/heads/main/data/originais/morbidade/desagregado/2020/A102812189_28_143_208.csv",
        "https://github.com/alexlopespereira/mba_enap/raw/refs/heads/main/data/originais/morbidade/desagregado/2020/A102850189_28_143_208.csv",
        "https://github.com/alexlopespereira/mba_enap/raw/refs/heads/main/data/originais/morbidade/desagregado/2020/A102927189_28_143_208.csv",
        "https://github.com/alexlopespereira/mba_enap/raw/refs/heads/main/data/originais/morbidade/desagregado/2020/A103139189_28_143_208.csv",
        "https://github.com/alexlopespereira/mba_enap/raw/refs/heads/main/data/originais/morbidade/desagregado/2020/A103238189_28_143_208.csv",
        "https://github.com/alexlopespereira/mba_enap/raw/refs/heads/main/data/originais/morbidade/desagregado/2020/A212148189_28_143_208.csv",
        "https://github.com/alexlopespereira/mba_enap/raw/refs/heads/main/data/originais/morbidade/desagregado/2020/A212323189_28_143_208.csv",
        "https://github.com/alexlopespereira/mba_enap/raw/refs/heads/main/data/originais/morbidade/desagregado/2020/A212334189_28_143_208.csv",
        "https://github.com/alexlopespereira/mba_enap/raw/refs/heads/main/data/originais/morbidade/desagregado/2021//A212345189_28_143_208.csv",
        "https://github.com/alexlopespereira/mba_enap/raw/refs/heads/main/data/originais/morbidade/desagregado/2021/A152003189_28_143_208.csv",
        "https://github.com/alexlopespereira/mba_enap/raw/refs/heads/main/data/originais/morbidade/desagregado/2021/A152053189_28_143_208.csv",
        "https://github.com/alexlopespereira/mba_enap/raw/refs/heads/main/data/originais/morbidade/desagregado/2021/A152118189_28_143_208.csv",
        "https://github.com/alexlopespereira/mba_enap/raw/refs/heads/main/data/originais/morbidade/desagregado/2021/A152142189_28_143_208.csv",
        "https://github.com/alexlopespereira/mba_enap/raw/refs/heads/main/data/originais/morbidade/desagregado/2021/A152222189_28_143_208.csv"
    ]
    response = requests.get(urls)
    response.raise_for_status()  # Raise an exception for bad status codes
    from io import BytesIO
    from zipfile import ZipFile
    import re

    with ZipFile(BytesIO(response.content)) as z:
        all_dfs = []
        for filename in z.namelist():
            if filename.endswith(".csv"):
                with z.open(filename) as f:
                    try:
                        df = pd.read_csv(f, encoding='iso8859-1', sep=';', skiprows=3, skipfooter=7, engine='python')
                        # Extract month and year
                        with z.open(filename) as file:
                            lines = file.readlines()
                            if len(lines) >= 3:
                                third_line = lines[2].decode('iso8859-1').strip()
                                match = re.search(r"Período:([a-zA-Z]{3})/(\d{4})", third_line)
                                if match:
                                    month_abbr = match.group(1)
                                    year = int(match.group(2))
                                    month_map = {
                                        "Jan": 1, "Fev": 2, "Mar": 3, "Abr": 4, "Mai": 5, "Jun": 6,
                                        "Jul": 7, "Ago": 8, "Set": 9, "Out": 10, "Nov": 11, "Dez": 12
                                    }
                                    month = month_map.get(month_abbr)

                                    df['mes'] = month
                                    df['ano'] = year

                                    df['Data'] = pd.to_datetime({'year': df['ano'], 'month': df['mes'], 'day': 1})
                                    all_dfs.append(df)
                            else:
                                print(f"File {filename} has less than 3 lines.")
                    except pd.errors.ParserError:
                        print(f"Error parsing file: {filename}")

    combined_df = pd.concat(all_dfs, ignore_index=True)

    sample_rows = [
        {
            # First row to check
            "filter": {
                "Município": "110001 Alta Floresta D'Oeste", "Data": "2019-12-01"
            },
            "expected_values": {
                "Internações": 193
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
        groundtruth_df=combined_df,
        input_value=urls,
        function_id="A6-E8",
        testcase_id="1",
        sample_rows=sample_rows,
        row_match_threshold=0.01
    )

    return test_case


def merge_pib_pop():
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





if __name__ == "__main__":
    # Example usage
    test_case = merge_pib_pop() #create_grupo()
    print("Generated Test Case:")
    print(json.dumps(test_case).replace("\n",""))