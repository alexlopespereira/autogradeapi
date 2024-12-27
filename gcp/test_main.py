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
def gerar_permutacao_em_grupos(names_list, seed, n):
    import math
    import pandas as pd
    import random
    
    # Set the random seed
    random.seed(seed)
    
    # Calculate number of people and groups needed
    N = len(names_list)
    num_groups = math.ceil(N/n)
    
    # Create list of group names repeated n times
    groups = ['Grupo ' + str(g) for g in range(num_groups)] * n
    groups.sort()
    groups = groups[:N]  # Trim to match number of names
    
    # Shuffle the names
    shuffled_names = names_list.copy()  # Create a copy to avoid modifying original list
    random.shuffle(shuffled_names)
    
    # Create DataFrame with explicit columns
    df = pd.DataFrame({
        'Grupo': groups,
        'Nome': shuffled_names
    })
    
    return df
    """

    payload = {
        "code": test_code,
        "function_id": "A6-E2"
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