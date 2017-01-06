import os
import sys
import subprocess
from random import random
import tempfile
from nalaf.learning.taggers import RelationExtractor
from nalaf import print_warning, print_verbose, print_debug, is_verbose_mode
import numpy
import scipy
from sklearn import svm


class SklSVM(RelationExtractor):
    """
    Base class to interact with [scikit-learn SVM](http://scikit-learn.org/stable/modules/svm.html#svm),
    concretly [SVC class](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC)
    This is actually based on [LIBSVM](https://www.csie.ntu.edu.tw/%7Ecjlin/libsvm/).
    """
    # Likely we want to create a hierarchy for ML "Trainers" or similar

    def __init__(self, model_path=None, classification_threshold=0.0, use_tree_kernel=False, **svc_parameters):
        assert not use_tree_kernel, NotImplementedError

        self.model_path = model_path if model_path is not None else tempfile.NamedTemporaryFile().name
        """the model (path) to read from / write to"""
        print_debug("SVM-Light model file path: " + self.model_path)

        self.classification_threshold = classification_threshold
        self.verbosity_level = str(0)  # for now, for verbosity=0; -- alternative: str(1 if is_verbose_mode else 0)
        self.svc_parameters = svc_parameters
        self.model = svm.SVC(**svc_parameters)
        self.feature_set = None

    def train(self, corpus, feature_set):
        self.feature_set = feature_set
        X, y = __class__._convert_edges_to_SVC_instances(corpus, feature_set)
        self.model.fit(X, y)
        print_debug(self.model.get_params())
        return self

    def annotate(self, corpus):
        X, y = __class__._convert_edges_to_SVC_instances(corpus, self.feature_set)
        y_pred = self.model.predict(X)
        y_size = len(y)
        print_debug("Mean accuracy: {}".format(sum(real == pred for real, pred in zip(y, y_pred)) / y_size))

        for edge, target_pred in zip(corpus.edges(), y_pred):
            edge.target = target_pred

        return corpus.form_predicted_relations()

    @staticmethod
    def _convert_edges_to_SVC_instances(corpus, feature_set):
        """
        rtype: Tuple[scipy.csr_matrix, List[int]]
        """
        num_edges = sum(1 for _ in corpus.edges())
        num_features = len(feature_set)

        # We first construct the X matrix of features with the sparse lil_matrix, which is efficient in reshaping its structure dynamically
        # At the end, we convert this to csr_matrix, which is efficient for algebra operations
        # See https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.sparse.lil_matrix.html#scipy.sparse.lil_matrix
        # See http://scikit-learn.org/stable/modules/svm.html#svm
        X = scipy.sparse.lil_matrix((num_edges, num_features), dtype=numpy.float64)
        y = numpy.zeros(num_edges, order='C')  # -- see: http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC

        allowed_features_keys = set(feature_set.values())

        for edge_index, edge in enumerate(corpus.edges()):
            for f_key in edge.features.keys():
                if f_key in allowed_features_keys:
                    value = edge.features[f_key]
                    f_index = f_key - 1
                    X[edge_index, f_index] = value

            y[edge_index] = edge.target

        print_debug("#instances: {}: #positive: {} vs. #negative: {}".format(num_edges, sum(v > 0 for v in y), sum(v < 0 for v in y)))

        X = X.tocsr()

        return (X, y)
