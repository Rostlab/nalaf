"""
Floyd Warshall graph algorithm for calculating dependency parse graph features
The method here avoids the use of an external library.
"""

def floyd_warshall(graph):
    """
    calculate the shortest path between two tokens using the Floyd Warshall
    algorithm where the graph is the dependency graph

    :param node_from: the token from which path must be calculated
    :type node_from: int
    :param node_to: the token to which path must be calculated
    :type node_to: int
    :param adjacency: a list of tokens in the sentence
    :type sentence: list[nalaf.structures.data.Token]
    :return: shortest path between each node
    :rtype: dict[list[int]]
    """

    dist = {}
    pred = {}
    for u in graph:
        dist[u] = {}
        pred[u] = {}
        for v in graph:
            dist[u][v] = float('inf')
            pred[u][v] = -1
        dist[u][u] = 0
        for neighbor in graph[u]:
            dist[u][neighbor] = graph[u][neighbor]
            pred[u][neighbor] = u

    for t in graph:
        # given dist u to v, check if path u - t - v is shorter
        for u in graph:
            for v in graph:
                newdist = dist[u][t] + dist[t][v]
                if newdist < dist[u][v]:
                    dist[u][v] = newdist
                    pred[u][v] = pred[t][v]  # route new path through t

    return dist, pred


def convert_to_dependency_graph(sentence):
    """
    helper function for the Floyd Warshall algorithm where it computes the
    adjacency matrix based on the dependency graph and sentence tokens
    """
    graph = {}

    for token in sentence:
        graph[token.features['id']] = {}

    for token in sentence:
        graph[token.features['id']] = {}
        for tok in token.features['dependency_to']:
            graph[token.features['id']][tok[0].features['id']] = 1
            graph[tok[0].features['id']][token.features['id']] = 1

    return graph


def get_path(token_from, token_to, part, sentence_id, graphs):
    """
    graphs is a mutable dictionary
    """

    sentence = part.sentences[sentence_id]
    if part.text not in graphs.keys():
        graphs[part.text] = {}
    if sentence_id not in graphs[part.text].keys():
        graphs[part.text][sentence_id] = {}
        graph = convert_to_dependency_graph(sentence)
        distances, parents = floyd_warshall(graph)
        graphs[part.text][sentence_id]['graph'] = graph
        graphs[part.text][sentence_id]['distances'] = distances
        graphs[part.text][sentence_id]['parents'] = parents
    else:
        distances = graphs[part.text][sentence_id]['distances']
        parents = graphs[part.text][sentence_id]['parents']

    path = []

    if token_from.features['id'] == token_to.features['id']:
        return path

    path.append(token_to)
    i, j = token_from.features['id'], token_to.features['id']

    while(True):
        parent = parents[i][j]
        if distances[i][j] == float('inf'):
            path = []
            return path
        path.append(sentence[parent])
        if parent == i:
            break
        j = parent

    path.reverse()
    return path


def build_walks(path, first_id=0, second_id=1):

    if len(path) != second_id + 1:
        ret_walks = build_walks(path, first_id=first_id + 1, second_id=second_id + 1)
    else:
        ret_walks = []

    all_walks = []

    forward_dep = []
    for dep in path[first_id].features['dependency_to']:
        if dep[0] == path[second_id]:
            forward_dep.append(dep)

    for f_dep in forward_dep:
        if len(ret_walks) > 0:
            for walk in ret_walks:
                new_walk = []
                new_walk.append(f_dep)
                for dep in walk:
                    new_walk.append(dep)
                all_walks.append(new_walk)
        else:
            new_walk = []
            new_walk.append(f_dep)
            all_walks.append(new_walk)

    backward_dep = []
    for dep in path[second_id].features['dependency_to']:
        if dep[0] == path[first_id]:
            backward_dep.append(dep)
    for b_dep in backward_dep:
        if len(ret_walks) > 0:
            for walk in ret_walks:
                new_walk = []
                new_walk.append(b_dep)
                for dep in walk:
                    new_walk.append(dep)
                all_walks.append(new_walk)
        else:
            new_walk = []
            new_walk.append(b_dep)
            all_walks.append(new_walk)

    return all_walks
