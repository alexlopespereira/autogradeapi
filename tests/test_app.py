


import json

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


floyd_warshall(*[[[0, 4, 5, "inf"], ["inf", 0, 2, 6], ["inf", "inf", 0, 3], ["inf", "inf", "inf", 0]], ["A", "D"]])