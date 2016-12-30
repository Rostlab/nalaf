
import numpy
import itertools

"""
Floyd-Warshall graph algorithm to compute the shortest paths between the dependency graphs of sentences.
See: https://en.wikipedia.org/wiki/Floyd–Warshall_algorithm

As of now, matrises are written fully. An obvious performance improvement is to write them sparsely.
"""

def compute_shortest_paths(sentence):
    """
    Compute the shortest paths of a sentence's dependency graph.

    Returns tuple: (dist, then)
        dist: matrix of minimal distances between pairs of tokens u, v
        then: matrix of

    To get then the path of a sentence's pair of tokens, use the method `path`
    """
    return floyd_warshall_with_path_reconstruction(sentence_to_weight_matrix(sentence))


def path(u, v, dist, then, sentence):
    if numpy.isnan(then[u, v]):
        return []
    else:
        try:
            path = [sentence[u]]
            while u != v:
                u = int(then[u, v])
                path.append(sentence[u])
            return path
        except:
            print(dist, then, sentence)
            raise


def floyd_warshall_with_path_reconstruction(weight):
    """
    Compute the shortest distances and paths in a graph matrix representation.

    Implementation of https://en.wikipedia.org/wiki/Floyd–Warshall_algorithm#Path_reconstruction

    matrix 'then' is the equivalent of 'next'
    """
    V = len(weight)
    dist = numpy.full([V, V], numpy.inf)
    then = numpy.full([V, V], numpy.nan)

    # Init
    for u in range(V):
        for v in range(V):
            dist[u, v] = weight[u, v]
            if weight[u, v] <= 1:
                then[u, v] = v

    # Dynamic Recursive
    for k in range(V):
        for i in range(V):
            for j in range(V):
                dist_i_k_j = dist[i, k] + dist[k, j]
                if dist[i, j] > dist_i_k_j:
                    dist[i, j] = dist_i_k_j
                    then[i, j] = then[i, k]

    return dist, then


def sentence_to_weight_matrix(sentence):
    """
    Converts the dependency graph of a sentence of tokens into a weight matrix.
    weight[u, v] = 0 iff u == v
    weight[u, v] = 1 iff u != v and are_bidirectionaly_directly_connected(u, v) == True
    weight[u, v] = 0 else
    """

    V = len(sentence)
    weight = numpy.full([V, V], numpy.inf)

    for from_token in sentence:
        for to_token, _ in from_token.features['dependency_to']:
            u = from_token.features['id']
            v = to_token.features['id']

            weight[u, v] = 1
            weight[v, u] = 1

    for u in range(V):
        weight[u, u] = 0

    return weight


class Path:

    def __init__(tokens):
        self.tokens = tokens
        self.nodes = [PathNode(t, None, None) for t in tokens]

        if len(self.tokens) == 0:
            self.head = self.last = None
            self.middle = []
        else:
            self.head = self.nodes[0]
            self.middle = self.nodes[1:-1]
            self.last = self.nodes[-1]

    def exists(self):
        return self.head is not None

    def __str__(self):
        return str_full()

    def str_full(self, token_to_string_fun=lambda token: token.word):
        return ' '.join(itertools.chain([self.head.str_full(lambda _: "")], (n.str_full() for n in self.middle)))

    def str_token_only(self):
        return ' '.join(n.str_token_only() for n in self.middle)

    def str_undirected_edge_only(self):
        return ' '.join(n.str_undirected_edge_only for n in ([self.head] + self.middle))

    def str_directed_edge_only(self):
        return ' '.join(n.str_directed_edge_only for n in ([self.head] + self.middle))


class PathNode:

    def __init__(token, edge_type, is_forward):
        self.token = token
        self.edge_type = edge_type
        self.is_forward = is_forward

    def __str__(self):
        return self.str_full()

    def str_full(self, token_to_string_fun=lambda token: token.word):
        str_token = self.str_token_only(token_to_string_fun)
        str_token = str_token + " " if str_token else ""

        return "{}{} {}".format(str_token, edge_type, self.str_direction())

    def str_token_only(self, token_to_string_fun):
        return "{}".format(token_to_string_fun(token))

    def str_undirected_edge_only(self):
        return "{}".format(edge_type)

    def str_directed_edge_only(self):
        return "{} {}".format(edge_type, self.str_direction())

    def str_direction(self):
        return "F" if self.is_forward else "B"
