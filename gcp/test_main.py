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
def floyd_warshall(adjacency_matrix, path):
    # Converter valores "inf" da matriz de adjacência para float('inf')
    for i in range(len(adjacency_matrix)):
        for j in range(len(adjacency_matrix[i])):
            if adjacency_matrix[i][j] == "inf":
                adjacency_matrix[i][j] = float('inf')
    
    num_vertices = len(adjacency_matrix)
    
    # Inicialização das matrizes de distâncias e próximos nós
    distance = [[float('inf')] * num_vertices for _ in range(num_vertices)]
    next_node = [[None] * num_vertices for _ in range(num_vertices)]
    
    for i in range(num_vertices):
        for j in range(num_vertices):
            distance[i][j] = adjacency_matrix[i][j]
            if adjacency_matrix[i][j] != float('inf') and i != j:
                next_node[i][j] = j

    # Atualização das distâncias com Floyd-Warshall
    for k in range(num_vertices):
        for i in range(num_vertices):
            for j in range(num_vertices):
                if distance[i][k] + distance[k][j] < distance[i][j]:
                    distance[i][j] = distance[i][k] + distance[k][j]
                    next_node[i][j] = next_node[i][k]
    
    # Função para reconstruir o caminho entre dois nós
    def reconstruct_path(start, end):
        if next_node[start][end] is None:
            return []
        path = [start]
        while start != end:
            start = next_node[start][end]
            path.append(start)
        return path

    start_node, end_node = path
    total_distance = distance[start_node][end_node]
    
    # Reconstruir caminho
    reconstructed_path = reconstruct_path(start_node, end_node)
    
    return [total_distance, reconstructed_path]
"""

    payload = {
        "code": test_code,
        "function_id": "A4-E10"
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