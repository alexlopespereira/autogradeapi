import json
from typing import List, Dict, Any

import numpy as np
import pandas as pd


def create_dataframe_test_case_flexible(
        groundtruth_df: pd.DataFrame,
        input_value: Any,
        function_id: str,
        testcase_id: str,
        sample_conditions: List[Dict[str, Any]],
        round_decimals: int = 1,
        row_match_threshold: float = 0.1
) -> Dict[str, Any]:
    """
    Create a test case specification with flexible sample row selection.

    Args:
        groundtruth_df: The reference DataFrame containing the expected output
        input_value: The input value(s) for the test case
        function_id: The function identifier
        testcase_id: The test case identifier
        sample_conditions: List of dictionaries containing filter conditions for sample rows
                         Each dict should have:
                         - 'filter_cols': columns to use in filter (optional)
                         - 'value_cols': columns to check values for (optional)
                         - 'conditions': dict of column-value pairs for filtering
        round_decimals: Number of decimal places for rounding
        row_match_threshold: Threshold for numeric comparisons

    Returns:
        Dict containing the complete test case specification
    """
    # Get column information
    columns = list(groundtruth_df.columns)
    dtypes = {col: str(groundtruth_df[col].dtype) for col in columns}

    # Create sample rows specifications
    sample_rows = []
    for condition_spec in sample_conditions:
        conditions = condition_spec['conditions']
        filter_cols = condition_spec.get('filter_cols', list(conditions.keys()))

        # If no specific value columns are provided, use all non-filter columns
        value_cols = condition_spec.get('value_cols', [col for col in columns if col not in filter_cols])

        # Find matching rows
        matching_rows = groundtruth_df
        for col, value in conditions.items():
            matching_rows = matching_rows[matching_rows[col] == value]

        if matching_rows.empty:
            raise ValueError(f"No rows match the conditions: {conditions}")

        # Take the first matching row
        row = matching_rows.iloc[0]

        # Create filter and expected values
        filter_dict = {col: conditions[col] for col in filter_cols}
        expected_values = {col: row[col] for col in value_cols if col in row}

        sample_rows.append({
            "filter": filter_dict,
            "expected_values": expected_values
        })

    # Calculate aggregate statistics
    aggregation_checks = {
        "total_rows": {
            "min": len(groundtruth_df),
            "max": len(groundtruth_df)
        },
        "sum": {},
        "mean": {}
    }

    # Add sum and mean for numeric columns with specified rounding
    numeric_cols = groundtruth_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        # Round sum and mean to specified decimal places
        col_sum = float(groundtruth_df[col].sum())
        col_mean = float(groundtruth_df[col].mean())

        aggregation_checks["sum"][col] = round(col_sum, round_decimals)
        aggregation_checks["mean"][col] = round(col_mean, round_decimals)

    # Create the complete test case dictionary
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


# Example usage function
def example_usage_flexible(file_path):
    # Create sample DataFrame
    df = pd.read_excel(file_path, sheet_name="MUNICÍPIOS", skiprows=1, converters={'COD. UF': str, 'COD. MUNIC': str})
    df = df.rename(columns={df.columns[-2]: "POPULACAO",
                            df.columns[-3]: "MUNICIPIO"})
    df = df.iloc[:, :-1]  # Remove a última coluna
    df['cod_ibge7'] = df['COD. UF'] + df['COD. MUNIC']
    df = df[:-40]


    # Define sample conditions
    sample_conditions = [
        {
            'filter_cols': [],
            'value_cols': [],
            'conditions': {'UF': 'DF'}
        },
        {
            'filter_cols': [],
            'value_cols': [],
            'conditions': {'cod_ibge7': '1100015'}
        }
    ]

    # Create test case
    test_case = create_dataframe_test_case_flexible(
        groundtruth_df=df,
        input_value=file_path,
        function_id="A6-E3",
        testcase_id="1",
        sample_conditions=sample_conditions
    )

    return test_case


if __name__ == "__main__":
    # Example usage
    test_case = example_usage_flexible('https://github.com/alexlopespereira/mba_enap/raw/main/data/originais/populacao/POP2024_20241101.xls')
    print("Generated Test Case:")
    print(json.dumps(test_case).replace("\n",""))