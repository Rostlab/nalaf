
import numpy
import itertools


def compute_shortest_path(sentence, token_1, token_2):
    """
    Compute the shortest path between the given pair of tokens considering the sentence's dependency graph.

    The method currently uses the Floyd Warshall algorithm to compute the shortest paths between all nodes in the graph
    and runs in O(V^3) time. Efforts are under way to reduce this.

    Returns Path
    """
    source = token_1.features['id']
    target = token_2.features['id']
    _, prev = dijkstra_original(source, target, sentence)
    return path_reversed(source, target, prev, sentence)


def compute_shortest_paths(sentence):
    """
    Compute the shortest paths between all pairs of the sentence's tokens considering the sentence's dependency graph.

    Returns tuple: (dist, then)
        dist: matrix of minimal distances between pairs of tokens u, v
        then: matrix of

    To get then the path of a sentence's pair of tokens, use the method `path`
    """
    return floyd_warshall_with_path_reconstruction(sentence_to_weight_matrix(sentence))


def path(u, v, then, sentence):
    """
    Traces back the path between tokens `u` and `v` after running `compute_shortest_paths`.

    Returns Path.
    """

    if numpy.isnan(then[u, v]):
        return Path([])
    else:
        tokens_path = [sentence[u]]
        while u != v:
            u = int(then[u, v])
            tokens_path.append(sentence[u])
        return Path(tokens_path)


def path_reversed(source, target, prev, sentence):
    """
    Traces back the path (reversedly) between tokens `source` and `target` after running `dijkstra_original`.

    Returns Path.
    """
    tokens_path = []
    u = target
    while u != source:
        tokens_path.append(u)
        u = prev[u]
        if numpy.isnan(u):  # no possible path
            return Path([])
        else:
            u = int(u)
    tokens_path.append(u)

    return Path(list([sentence[t] for t in reversed(tokens_path)]))


def dijkstra_original(source, target, sentence, weight=None):
    """
    Computes the shortest path between tokens `u` and `v` with the original Dijkstra algorithm, O(V^2).

    The implementation sort of follows the pseudocode in
    https://en.wikipedia.org/w/index.php?title=Dijkstra%27s_algorithm&oldid=757046675#Pseudocode

    Returns Path
    """

    if weight is None:
        weight = sentence_to_weight_matrix(sentence)

    unvisited_set = set()

    V = len(weight)
    dist = numpy.full([V], numpy.inf)
    prev = numpy.full([V], numpy.nan)

    def are_neighbors(u, v):
        return not numpy.isinf(weight[u, v])

    # Init
    for v in range(V):
        dist[v] = weight[source, v]
        unvisited_set.update({v})
        if are_neighbors(source, v):
            prev[v] = source

    unvisited_set.remove(source)

    # Dynamic Recursive
    while len(unvisited_set) > 0:
        u = kinda_argmin(unvisited_set, lambda u: dist[u], target)

        if u == target or u is None:  # when u is None, then there is no possible path
            break

        unvisited_set.remove(u)

        for v in range(V):
            if v in unvisited_set and are_neighbors(u, v):
                dist_source_u_v = dist[u] + weight[u, v]
                if dist[v] > dist_source_u_v:
                    dist[v] = dist_source_u_v
                    prev[v] = u

    return dist, prev


def kinda_argmin(iterable, key, target):
    minimum = float('inf')
    ret = None
    for x in iterable:
        x_val = key(x)
        x_is_closer_to_target = ret is None or (abs(target - x) < abs(target - ret))
        if x_is_closer_to_target and x_val < minimum:
            ret = x
    return ret


def floyd_warshall_with_path_reconstruction(weight):
    """
    Compute the shortest distances and paths in a graph matrix representation as per the Floyd-Warshall algorithm.

    Implementation of https://en.wikipedia.org/wiki/Floyd–Warshall_algorithm#Path_reconstruction

    matrix 'then' is the equivalent of 'next'
    """
    # MAYBE As of now, matrises are written fully. An obvious performance improvement is to write them sparsely.

    V = len(weight)
    dist = numpy.full([V, V], numpy.inf)
    then = numpy.full([V, V], numpy.nan)

    # Init
    for u in range(V):
        for v in range(V):
            dist[u, v] = weight[u, v]
            if not numpy.isinf(weight[u, v]):
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

    __NODE_SEPARATOR = " ~ "

    def __init__(self, tokens):
        self.tokens = tokens
        self.nodes = []

        for u_token, v_token in zip(tokens, tokens[1:]):  # Note: the last one is not added yet, see below
            u_dep_from = u_token.features['dependency_from']
            v_dep_from = v_token.features['dependency_from']

            if v_dep_from is not None and v_dep_from[0] == u_token:
                edge_type = v_dep_from[1]
                is_forward = True
            elif u_dep_from is not None and u_dep_from[0] == v_token:
                edge_type = u_dep_from[1]
                is_forward = False
            else:
                raise AssertionError(("One must be a dependency of each other", u_token, v_token, u_dep_from, v_dep_from, tokens))

            self.nodes.append(PathNode(u_token, edge_type, is_forward))

        if len(self.tokens) == 0:
            self.head = self.last = []
            self.middle = []
        else:
            self.nodes.append(PathNode(self.tokens[-1], edge_type="", is_forward=None))  # last one
            self.head = [self.nodes[0]]
            self.middle = self.nodes[1:-1]
            self.last = [self.nodes[-1]]

    def exists(self):
        return len(self.head) >= 2

    def __str__(self):
        return self.str_full()

    def __repr__(self):
        return __class__.__NODE_SEPARATOR.join(repr(n) for n in self.nodes)

    def __eq__(self, other):
        return self.nodes == other.nodes

    def str_full(self, token_to_string_fun=lambda token: token.word):
        return __class__.__NODE_SEPARATOR.join(itertools.chain(
            (head.str_full(lambda _: "") for head in self.head),
            (n.str_full(token_to_string_fun) for n in self.middle)))

    def str_token_only(self):
        return __class__.__NODE_SEPARATOR.join(n.str_token_only() for n in self.middle)

    def str_undirected_edge_only(self):
        return __class__.__NODE_SEPARATOR.join(n.str_undirected_edge_only for n in (self.head + self.middle))

    def str_directed_edge_only(self):
        return __class__.__NODE_SEPARATOR.join(n.str_directed_edge_only for n in (self.head + self.middle))


class PathNode:

    def __init__(self, token, edge_type, is_forward):
        self.token = token
        self.edge_type = edge_type
        self.is_forward = is_forward

    def __str__(self):
        return self.str_full()

    def __repr__(self):
        return str((self.token.word, self.edge_type, self.is_forward))

    def __eq__(self, other):
        return (self.token == other.token and
                self.edge_type == other.edge_type and
                self.is_forward == other.is_forward)

    def str_full(self, token_to_string_fun=lambda token: token.word):
        return ' '.join(filter(None, [self.str_token_only(token_to_string_fun), self.edge_type, self.str_direction()]))

    def str_token_only(self, token_to_string_fun):
        return token_to_string_fun(self.token)

    def str_undirected_edge_only(self):
        return self.edge_type

    def str_directed_edge_only(self):
        return ' '.join(filter(None, [edge_type, self.str_direction()]))

    def str_direction(self):
        return "" if (self.is_forward is None or not self.edge_type) else ("F" if self.is_forward else "B")
