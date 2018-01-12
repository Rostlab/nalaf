import numpy
import itertools
from nalaf import print_debug


def compute_shortest_path(sentence, token_1_index, token_2_index):
    """
    Compute the shortest path between the given pair of tokens considering the sentence's dependency graph.

    The method uses the Dijkstra algorithm internally.

    Returns Path
    """
    if sentence and not sentence[0].features.get("tmp_id"):
        for index, t in enumerate(sentence):
            t.features['tmp_id'] = index  # See Edge::get_combined_sentence

    source = token_1_index
    target = token_2_index
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
    if sentence and not sentence[0].features.get("tmp_id"):
        for index, t in enumerate(sentence):
            t.features['tmp_id'] = index  # See Edge::get_combined_sentence

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
        for to_token, _ in (from_token.features['dependency_to'] + from_token.features['user_dependency_to']):
            u = from_token.features['tmp_id']
            v = to_token.features['tmp_id']

            weight[u, v] = 1
            weight[v, u] = 1

    for u in range(V):
        weight[u, u] = 0

    return weight


class Path:

    __STR_NODE_SEPARATOR = " ~~ "
    __STR_SOURCE = "[SOURCE"
    __STR_TARGET = "TARGET]"

    def __init__(self, tokens, name="", is_edge_type_constant=False, there_is_target=True, default_n_grams=None):
        self.tokens = tokens
        self.nodes = []
        self.name = name
        self.is_edge_type_constant = is_edge_type_constant
        self.default_n_grams = default_n_grams if default_n_grams is not None else []

        for u_token, v_token in zip(tokens, tokens[1:]):  # Note: the last one is not added yet, see below
            if is_edge_type_constant:
                edge_type = ""
                is_forward = None

            else:
                parser_defined = __class__._get_dep_edges(
                    u_token, v_token,
                    __class__.__mk_list_rm_None(u_token.features['dependency_from']),
                    __class__.__mk_list_rm_None(v_token.features['dependency_from']))

                user_defined = __class__._get_dep_edges(
                    u_token, v_token,
                    u_token.features['user_dependency_from'],
                    v_token.features['user_dependency_from'])

                all_dep_edges = parser_defined + user_defined

                assert len(all_dep_edges) > 0, \
                    ("One must be a dependency of the other", u_token, v_token, tokens)

                if len(all_dep_edges) > 1:
                    print_debug("Multiple dependencies are not handled yet; defaulted to first. This should strictly only happen with user-defined dependencies")

                edge_type = all_dep_edges[0][0]
                is_forward = all_dep_edges[0][1]

            self.nodes.append(PathNode(u_token, edge_type, is_forward))

        if len(self.tokens) == 0:
            self.exists = False
            self.source = self.target = self.middle = []
        else:
            self.exists = True

            self.nodes.append(PathNode(self.tokens[-1], edge_type="", is_forward=None, is_target=there_is_target))
            self.nodes[0].is_source = True

            self.source = [self.nodes[+0]]

            if there_is_target:
                self.middle = self.nodes[1:-1]
                self.target = [self.nodes[-1]]
            else:
                self.middle = self.nodes[1:]
                self.target = []

    @staticmethod
    def __mk_list_rm_None(item):
        return list(filter(None.__ne__, [item]))

    @staticmethod
    def _get_dep_edges(u_token, v_token, u_dep_froms, v_dep_froms):
        edges = []  # list of tuples with (edge_type: str, is_forward: Bool)

        for dep_token, edge_type in v_dep_froms:
            if dep_token == u_token:
                is_forward = True
                edges.append((edge_type, is_forward))

        for dep_token, edge_type in u_dep_froms:
            if dep_token == v_token:
                is_forward = False
                edges.append((edge_type, is_forward))

        return edges

    def change_name(self, new_name):
        self.name = new_name
        return self

    def change_default_n_grams(self, default_n_grams):
        self.default_n_grams = default_n_grams
        return self

    def __str_join_nodes(self, nodes_strs):
        return "<" + __class__.__STR_NODE_SEPARATOR.join(nodes_strs) + ">"

    def __str_token(self, node, middle, source, target):
        if node.is_source:
            return source
        elif node.is_target:
            return target
        else:
            assert node.is_middle()
            return middle

    def __str__(self):
        return self.str_full()

    def __repr__(self):
        return self.__str_join_nodes(repr(n) for n in self.nodes)

    def __eq__(self, other):
        return self.nodes == other.nodes

    def str_full(
        self,
        str_middle_token=lambda t: t.word,
        str_source_token=lambda t: __class__.__STR_SOURCE,
        str_target_token=lambda t: __class__.__STR_TARGET,
    ):
        l = (n.str_full(self.__str_token(n, str_middle_token, str_source_token, str_target_token)) for n in self.nodes)
        return self.__str_join_nodes(l)

    def str_token_only(self, str_middle_token=lambda t: t.word):
        return self.__str_join_nodes(n.str_token_only(str_middle_token) for n in self.middle)

    def str_undirected_edge_only(self):
        return self.__str_join_nodes(n.str_undirected_edge_only() for n in (self.source + self.middle))

    def str_directed_edge_only(self):
        return self.__str_join_nodes(n.str_directed_edge_only() for n in (self.source + self.middle))

    def __n_grams(self, iterable, n_gram):
        return zip(*(iterable[i:] for i in range(0, n_gram)))

    def strs_n_gram_undirected_edge_only(self, n_gram):
        nodes = self.source + self.middle
        for nodes_group in self.__n_grams(nodes, n_gram):
            yield self.__str_join_nodes(n.str_undirected_edge_only() for n in nodes_group)

    def strs_n_gram_directed_edge_only(self, n_gram):
        nodes = self.source + self.middle
        for nodes_group in self.__n_grams(nodes, n_gram):
            yield self.__str_join_nodes(n.str_directed_edge_only() for n in nodes_group)

    def strs_n_gram_token_only(
        self,
        n_gram,
        str_middle_token=lambda t: t.word,
        str_source_token=lambda t: __class__.__STR_SOURCE,
        str_target_token=lambda t: __class__.__STR_TARGET
    ):
        if n_gram == 1:
            nodes = self.middle
        else:
            nodes = self.nodes

        for nodes_group in self.__n_grams(nodes, n_gram):
            l = (n.str_token_only(self.__str_token(n, str_middle_token, str_source_token, str_target_token)) for n in nodes_group)
            yield self.__str_join_nodes(l)

    def strs_n_gram_full(
        self,
        n_gram,
        str_middle_token=lambda t: t.word,
        str_source_token=lambda t: __class__.__STR_SOURCE,
        str_target_token=lambda t: __class__.__STR_TARGET
    ):
        if n_gram == 1:
            nodes = self.source + self.middle
        else:
            nodes = self.nodes

        for nodes_group in self.__n_grams(nodes, n_gram):
            l = (n.str_full(self.__str_token(n, str_middle_token, str_source_token, str_target_token)) for n in nodes_group)
            yield self.__str_join_nodes(l)


class PathNode:

    def __init__(self, token, edge_type, is_forward, is_source=False, is_target=False):
        self.token = token
        self.edge_type = edge_type
        self.is_forward = is_forward
        self.is_source = is_source
        self.is_target = is_target

    def is_middle(self):
        return not (self.is_source or self.is_target)

    def __str__(self):
        return self.str_full()

    def __repr__(self):
        return str((self.token.word, self.edge_type, self.is_forward, self.is_source, self.is_target))

    def __eq__(self, other):
        return (self.token == other.token and
                self.edge_type == other.edge_type and
                self.is_forward == other.is_forward)

    def str_full(self, str_token=lambda t: t.word):
        return ' '.join(filter(None, [self.str_token_only(str_token), self.str_directed_edge_only()]))

    def str_token_only(self, str_token):
        return str_token(self.token)

    def str_undirected_edge_only(self):
        return self.edge_type

    def str_directed_edge_only(self):
        return '-'.join(filter(None, [self.edge_type, self.str_direction()]))

    def str_direction(self):
        return "" if (self.is_forward is None or not self.edge_type) else ("F" if self.is_forward else "B")
