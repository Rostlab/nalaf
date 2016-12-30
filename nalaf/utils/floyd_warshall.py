
import numpy

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
    import random
    return [random.random()]


def floyd_warshall_with_path_reconstruction(weight):
    """
    Compute the shortest distances and paths in a graph matrix representation.

    Implementation of https://en.wikipedia.org/wiki/Floyd–Warshall_algorithm#Path_reconstruction

    matrix 'then' is the equivalent of 'next'
    """
    V = len(weight)
    dist = numpy.full([V, V], numpy.inf)
    then = numpy.full([V, V], numpy.nan)

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

    # for from_token in sentence:
    #     for to_token, _ in from_token.features['dependency_to']:
    #         u = from_token.features['id']
    #         v = to_token.features['id']
    #
    #         weight[u, v] = 1
    #         weight[v, u] = 1

    # for u in V:
    #     weight[u, u] = 0

    return weight
