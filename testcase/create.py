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
        row_match_threshold: float = 0.1
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
                'Group': '2'  # How to find this row
            },
            "expected_values": { 'Name': 'Olga Crocker' }
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


if __name__ == "__main__":
    # Example usage
    test_case = create_grupo()
    print("Generated Test Case:")
    print(json.dumps(test_case).replace("\n",""))