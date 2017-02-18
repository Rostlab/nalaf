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
from sklearn.preprocessing import MaxAbsScaler
from sklearn.feature_selection import VarianceThreshold
import time
from sklearn.pipeline import make_pipeline


class SklSVM(RelationExtractor):
    """
    Base class to interact with [scikit-learn SVM](http://scikit-learn.org/stable/modules/svm.html#svm),
    concretly [SVC class](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC)
    This is actually based on [LIBSVM](https://www.csie.ntu.edu.tw/%7Ecjlin/libsvm/).
    """
    # Likely we want to create a hierarchy for ML "Trainers" or similar

    def __init__(self, model_path=None, classification_threshold=0.0, use_tree_kernel=False, preprocess=True, **svc_parameters):
        assert not use_tree_kernel, NotImplementedError
        assert classification_threshold == 0, NotImplementedError

        self.version = "1.0.0"

        if isinstance(preprocess, list):
            self.preprocess = make_pipeline(*preprocess)
        elif preprocess is True:
            self.preprocess = make_pipeline(
                # # See: http://scikit-learn.org/stable/modules/feature_selection.html#removing-features-with-low-variance
                # (lambda p: VarianceThreshold(threshold=(p * (1 - p))))(0.01),

                # See http://stackoverflow.com/a/41601532/341320
                # logtran = FunctionTransformer(snp.log1p, accept_sparse=True, validate=True)
                # X = logtran.transform(X)
                MaxAbsScaler(copy=False),
            )
        else:
            self.preprocess = make_pipeline(None)

        self.model = svm.SVC(**svc_parameters)

        self.global_feature_set = None
        """
        Dict[String, Int] : Map of feature names to feature keys of an assumed global featured corpus
        """
        self.allowed_feature_names = None
        """
        Set[String] : Set of allowed feature names, if given
        """
        self.allowed_feature_keys = None
        """
        Set[Int] : Set of allowed feature keys - If None, the allowed keys implictly equal the features' keys of the training data
        """
        self.final_allowed_feature_mapping = None
        """
        Dict[Int, Int] or Function[Int] -> Int.

        A map of allowed (feature keys) to (final feature indexes as written in the instances).

        The maximum value of the final index is equal or less than the maximum value of the original feature keys
        and only equal in case all feature keys are allowed. The final indexes are enumerated from 0 to n-1.

        That is, for example, say the corpus has features keys = {0, 1, 2, 3, 4}, yet the allowed feature indexes
        are only = {1, 3, 4}. The resulting final_allowed_feature_mapping is = {1: 0, 3: 1, 4: 2}.
        Note: the order is NOT respected nor checked.

        When they are equal, there is a 1:1 correspondence between keys and indexes, e.g. {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
        Note: The order IS respected.
        """

    def train(self, training_corpus):
        if self.final_allowed_feature_mapping is None:
            self.set_allowed_feature_keys_from_corpus(training_corpus)

        X, y = self.__convert_edges_features_to_vector_instances(training_corpus)
        print_debug("Train SVC with #samples {} - #features {} - params: {}".format(X.shape[0], X.shape[1], str(self.model.get_params())))
        start = time.time()
        self.model.fit(X, y)
        end = time.time()
        print_debug("SVC train, running time: ", (end - start))

        return self

    def annotate(self, corpus):
        X, y = self.__convert_edges_features_to_vector_instances(corpus)
        y_pred = self.model.predict(X)
        y_size = len(y)
        print_debug("Mean accuracy: {}".format(sum(real == pred for real, pred in zip(y, y_pred)) / y_size))

        for edge, target_pred in zip(corpus.edges(), y_pred):
            edge.pred_target = target_pred

        return corpus.form_predicted_relations()

    ##################################################################################################################

    def set_allow_all_feature_keys(self):
        """
        Generate a 1:1 feature key to final feature index mapping. The other IS respected.

        rtype: Tuple[None, None, Function[Int] -> Int]
        """
        self.final_allowed_feature_mapping = (lambda f_key: f_key)

        return self

    def set_allowed_feature_names(self, global_feature_set, allowed_feature_names):
        self.global_feature_set = global_feature_set
        self.allowed_feature_names = allowed_feature_names
        allowed_feature_keys = {self.global_feature_set[f_name] for f_name in self.allowed_feature_names if self.global_feature_set.get(f_name, None)}
        self.set_allowed_feature_keys(allowed_feature_keys)

        return self

    def set_allowed_feature_keys(self, allowed_feature_keys):
        """
        Generate a 1:1 feature key to final feature index mapping. The other IS respected.

        rtype: Tuple[None, Set[Int], Dict[Int, Int]]
        """
        self.allowed_feature_keys = set(allowed_feature_keys)
        self.final_allowed_feature_mapping = {}

        for allowed_feat_key in allowed_feature_keys:
            self.final_allowed_feature_mapping[allowed_feat_key] = len(self.final_allowed_feature_mapping)

        return self

    def set_allowed_feature_keys_from_corpus(self, corpus):
        """
        Generate the final mapping of features from the feature indexes that are present in the given corpus (considered a training one).

        rtype: Tuple[None, None, Set[Int], Dict[Int, Int]]
        """
        self.allowed_feature_keys = {f_key for edge in corpus.edges() for f_key in edge.features.keys()}
        self.final_allowed_feature_mapping = {}

        for allowed_feat_key in allowed_feature_keys:
            self.final_allowed_feature_mapping[allowed_feat_key] = len(self.final_allowed_feature_mapping)

        return self

    ##################################################################################################################

    def write_vector_instances(self, corpus, global_feature_set):
        self.global_feature_set = global_feature_set
        if self.final_allowed_feature_mapping is None:
            self.set_allow_all_feature_keys()
            num_features = len(global_feature_set)
        else:
            assert(isinstance(self.final_allowed_feature_mapping, dict))
            num_features = len(self.final_allowed_feature_mapping)

        X, y, groups = self.__gen_vector_instances(corpus, num_features=num_features)

        for edge in corpus.edges():
            assert(edge.initial_instance_index is not None)
            edge.features_vector = X.getrow(edge.initial_instance_index)

        return X, y, groups

    def __gen_vector_instances(self, corpus, num_features):

        if isinstance(self.final_allowed_feature_mapping, dict):
            final_allowed_feature_mapping_fun = (lambda f_key: self.final_allowed_feature_mapping.get(f_key, None))
        else:
            final_allowed_feature_mapping_fun = self.final_allowed_feature_mapping

        def fun(X, y, corpus):
            groups = {}
            instance_index = -1
            for docid, document in corpus.documents.items():
                groups[docid] = []  # Note, some documents may not generate instances -- Define them with empty lists

                for edge in document.edges():
                    instance_index += 1
                    edge.initial_instance_index = instance_index
                    groups[docid] = groups.get(docid, []) + [instance_index]

                    for f_key in edge.features.keys():

                        f_index = final_allowed_feature_mapping_fun(f_key)

                        if f_index is not None:
                            value = edge.features[f_key]
                            X[instance_index, f_index] = value

                    y[instance_index] = edge.real_target

            return X, y, groups

        return __class__._create_instances(
            num_features=num_features,
            corpus=corpus,
            preprocess=self.preprocess,
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

        X, y, groups = setting_function(X, y, corpus)

        X = X.tocsr()

        print_verbose("SVC before preprocessing, #features: {} && max value: {}".format(X.shape[1], max(sklearn.utils.sparsefuncs.min_max_axis(X, axis=0)[1])))
        if preprocess:
            X = preprocess.fit_transform(X)  # __class__._preprocess(X)
            print_debug("SVC after preprocessing, #features: {} && max value: {}".format(X.shape[1], max(sklearn.utils.sparsefuncs.min_max_axis(X, axis=0)[1])))

        end = time.time()
        print_debug("SVC convert instances, running time: ", (end - start))

        return (X, y, groups)


    def __convert_edges_features_to_vector_instances(self, corpus):
        if __class__._vector_instances_already_computed(corpus):
            return __class__._convert_edges_features_reusing_computed_vector_instances(corpus)
        else:
            return self.__gen_vector_instances(corpus, len(self.final_allowed_feature_mapping))


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

        X = X.tocsr()

        return (X, y)
