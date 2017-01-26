import os
import sys
import subprocess
from random import random
import tempfile
from nalaf.learning.taggers import RelationExtractor
from nalaf import print_warning, print_verbose, print_debug, is_verbose_mode
import numpy as np
import scipy
import sklearn
from sklearn import svm
from sklearn.preprocessing import FunctionTransformer, maxabs_scale
from sklearn.feature_selection import VarianceThreshold
import time


class SklSVM(RelationExtractor):
    """
    Base class to interact with [scikit-learn SVM](http://scikit-learn.org/stable/modules/svm.html#svm),
    concretly [SVC class](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC)
    This is actually based on [LIBSVM](https://www.csie.ntu.edu.tw/%7Ecjlin/libsvm/).
    """
    # Likely we want to create a hierarchy for ML "Trainers" or similar

    def __init__(self, model_path=None, classification_threshold=0.0, use_tree_kernel=False, preprocess=False, **svc_parameters):
        assert not use_tree_kernel, NotImplementedError

        self.model_path = model_path if model_path is not None else tempfile.NamedTemporaryFile().name
        """the model (path) to read from / write to"""
        print_debug("SVM-Light model file path: " + self.model_path)

        self.classification_threshold = classification_threshold
        self.preprocess = preprocess
        self.verbosity_level = str(0)  # for now, for verbosity=0; -- alternative: str(1 if is_verbose_mode else 0)
        self.svc_parameters = svc_parameters
        self.model = svm.SVC(**svc_parameters)
        self.global_feature_set = None
        self.allowed_features_keys = None
        self.final_allowed_key_mapping = None


    def train(self, training_corpus, global_feature_set):
        self.global_feature_set = global_feature_set  # The total/global feature set is actually not used
        self.allowed_features_keys, self.final_allowed_key_mapping = \
            __class__._gen_allowed_and_final_mapping_features_keys(training_corpus)

        X, y = __class__._convert_edges_features_to_vector_instances(training_corpus, self.preprocess, self.final_allowed_key_mapping)
        print_debug("Train SVC with #samples {} - #features {} - params: {}".format(X.shape[0], X.shape[1], str(self.model.get_params())))
        start = time.time()
        self.model.fit(X, y)
        end = time.time()
        print_debug("SVC train, running time: ", (end - start))
        return self


    def annotate(self, corpus):
        X, y = __class__._convert_edges_features_to_vector_instances(corpus, self.preprocess, self.final_allowed_key_mapping)
        y_pred = self.model.predict(X)
        y_size = len(y)
        print_debug("Mean accuracy: {}".format(sum(real == pred for real, pred in zip(y, y_pred)) / y_size))

        for edge, target_pred in zip(corpus.edges(), y_pred):
            edge.pred_target = target_pred

        return corpus.form_predicted_relations()


    def write_vector_instances(self, corpus, global_feature_set):
        self.global_feature_set = global_feature_set

        X, y = __class__._gen_vector_instances(corpus, global_feature_set, self.preprocess)

        for edge_index, edge in enumerate(corpus.edges()):
            edge.features_vector = X.getrow(edge_index)

        return X, y


    @staticmethod
    def _gen_vector_instances(corpus, global_feature_set, preprocess):

        def fun(X, y, corpus):
            for edge_index, edge in enumerate(corpus.edges()):
                for f_key in edge.features.keys():
                    f_index = f_key - 1  # f_keys are 1-indexed --> convert to 0-indexed
                    value = edge.features[f_key]
                    X[edge_index, f_index] = value

                y[edge_index] = edge.real_target

            return X, y


        return __class__._create_instances(
            num_features=len(global_feature_set),
            corpus=corpus,
            preprocess=preprocess,
            setting_function=fun
        )


    @staticmethod
    def _create_instances(num_features, corpus, preprocess, setting_function):
        """
        rtype: Tuple[scipy.sparse.csr_matrix, List[int]]
        """
        start = time.time()

        num_instances = sum(1 for _ in corpus.edges())

        # We first construct the X matrix of features with the sparse lil_matrix, which is efficient in reshaping its structure dynamically
        # At the end, we convert this to csr_matrix, which is efficient for algebra operations
        # See https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.sparse.lil_matrix.html#scipy.sparse.lil_matrix
        # See http://scikit-learn.org/stable/modules/svm.html#svm
        X = scipy.sparse.lil_matrix((num_instances, num_features), dtype=np.float64)
        y = np.zeros(num_instances, order='C')  # -- see: http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC

        X, y = setting_function(X, y, corpus)

        X = X.tocsr()

        print_verbose("SVC before preprocessing, #features: {} && max value: {}".format(X.shape[1], max(sklearn.utils.sparsefuncs.min_max_axis(X, axis=0)[1])))
        if preprocess:
            X = __class__._preprocess(X)
            print_verbose("SVC after preprocessing, #features: {} && max value: {}".format(X.shape[1], max(sklearn.utils.sparsefuncs.min_max_axis(X, axis=0)[1])))

        end = time.time()
        print_debug("SVC convert instances, running time: ", (end - start))

        return (X, y)


    @staticmethod
    def _gen_allowed_and_final_mapping_features_keys(corpus):
        """
        Generate the set and mapping of feature keys that are present in the given corpus (considered a training one).
        """
        allowed_keys = {fkey for edge in corpus.edges() for fkey in edge.features.keys()}
        final_mapping_keys = {}
        num_feat = 0
        for allowed_feat_key in allowed_keys:
            final_mapping_keys[allowed_feat_key] = num_feat
            num_feat += 1

        return (allowed_keys, final_mapping_keys)


    @staticmethod
    def _convert_edges_features_anew(corpus, preprocess, final_allowed_key_mapping):

        if final_allowed_key_mapping is None:
            _, final_allowed_key_mapping = __class__._gen_allowed_and_final_mapping_features_keys(corpus)

        def fun(X, y, corpus):
            for edge_index, edge in enumerate(corpus.edges()):
                for f_key in edge.features.keys():
                    if f_key in final_allowed_key_mapping:
                        f_index = final_allowed_key_mapping[f_key]
                        value = edge.features[f_key]
                        X[edge_index, f_index] = value

                y[edge_index] = edge.real_target

            return X, y


        return __class__._create_instances(
            num_features=len(final_allowed_key_mapping),
            corpus=corpus,
            preprocess=preprocess,
            setting_function=fun
        )


    @staticmethod
    def _convert_edges_features_to_vector_instances(corpus, preprocess, final_allowed_key_mapping=None):
        if __class__._vector_instances_already_computed(corpus):
            return __class__._convert_edges_features_reusing_computed_vector_instances(corpus)
        else:
            return __class__._convert_edges_features_anew(corpus, preprocess, final_allowed_key_mapping)


    @staticmethod
    def _vector_instances_already_computed(corpus):
        return next(corpus.edges()).features_vector is not None


    @staticmethod
    def _convert_edges_features_reusing_computed_vector_instances(corpus):
        num_instances = sum(1 for _ in corpus.edges())
        num_features = next(corpus.edges()).features_vector.shape[1]

        X = scipy.sparse.lil_matrix((num_instances, num_features), dtype=np.float64)
        y = np.zeros(num_instances, order='C')  # -- see: http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC

        for edge_index, edge in enumerate(corpus.edges()):
            X[edge_index, :] = edge.features_vector
            y[edge_index] = edge.real_target

        return (X, y)


    @staticmethod
    def _preprocess(X):
        # X = __class__._assure_min_variance(X)
        X = __class__._scale(X)
        return X


    @staticmethod
    def _assure_min_variance(X, p=0.99):
        # See: http://scikit-learn.org/stable/modules/feature_selection.html#removing-features-with-low-variance
        selector = VarianceThreshold(threshold=(p * (1 - p)))
        X = selector.fit_transform(X)
        return X


    @staticmethod
    def _scale(X):
        # See http://stackoverflow.com/a/41601532/341320
        logtran = FunctionTransformer(np.log1p, accept_sparse=True, validate=True)
        # X = logtran.transform(X)
        X = maxabs_scale(X, copy=False)
        return X
