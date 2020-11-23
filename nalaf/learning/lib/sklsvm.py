from nalaf.learning.taggers import RelationExtractor
from nalaf import print_debug
import numpy as np
import scipy
import sklearn
from sklearn import svm
from sklearn.preprocessing import MaxAbsScaler
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


    def train(self, training_corpus):
        X, y = self.__convert_edges_features_to_vector_instances(training_corpus)
        X = self.preprocess.fit_transform(X)
        print_debug("SVC after preprocessing, #features: {} && max value: {}".format(X.shape[1], max(sklearn.utils.sparsefuncs.min_max_axis(X, axis=0)[1])))

        print_debug("Train SVC with #samples {} - #features {} - params: {}".format(X.shape[0], X.shape[1], str(self.model.get_params())))
        start = time.time()
        self.model.fit(X, y)
        end = time.time()
        print_debug("SVC train, running time: ", (end - start))

        return self


    def annotate(self, corpus):
        X, y = self.__convert_edges_features_to_vector_instances(corpus)

        if X.shape[0] == 0:
            # no instances at all (corpus with no edges) --> nothing to do with the corpus
            return corpus

        else:
            X = self.preprocess.transform(X)
            print_debug("SVC after preprocessing, #features: {} && max value: {}".format(X.shape[1], max(sklearn.utils.sparsefuncs.min_max_axis(X, axis=0)[1])))

            # Pure classification prediction
            y_pred = self.model.predict(X)
            print_debug("Mean accuracy: {}".format(sum(real == pred for real, pred in zip(y, y_pred)) / len(y)))  # same as == self.model.score(X, y))

            for edge, target_pred in zip(corpus.edges(), y_pred):
                edge.pred_target = target_pred

            return corpus.form_predicted_relations()


    # ----------------------------------------------------------------------------------------------------

    def write_vector_instances(self, corpus, global_feature_set):
        num_features = len(global_feature_set)

        X, y, groups = self.__gen_vector_instances(corpus, num_features=num_features)

        for edge in corpus.edges():
            assert(edge.initial_instance_index is not None)
            edge.features_vector = X.getrow(edge.initial_instance_index)

        return X, y, groups

    def __gen_vector_instances(self, corpus, num_features):

        def fun(X, y, corpus):
            groups = {}
            instance_index = -1
            for docid, document in corpus.documents.items():
                groups[docid] = []  # Note, some documents may not generate instances -- Define them with empty lists

                for edge in document.edges():
                    instance_index += 1
                    edge.initial_instance_index = instance_index
                    groups[docid] = groups.get(docid, []) + [instance_index]

                    for f_index, value in edge.features.items():
                        X[instance_index, f_index] = value

                    y[instance_index] = edge.real_target

            return X, y, groups

        return __class__._create_instances(
            num_features=num_features,
            corpus=corpus,
            setting_function=fun
        )

    @staticmethod
    def _create_instances(num_features, corpus, setting_function):
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

        end = time.time()
        print_debug("SVC convert instances, running time: ", (end - start))

        return (X, y, groups)


    def __convert_edges_features_to_vector_instances(self, corpus):
        if __class__._are_vector_instances_already_computed(corpus):
            return __class__._convert_edges_features_reusing_computed_vector_instances(corpus)
        else:
            # + 1 since the keys are 0-indexed, that is a sole feature indexed by 0 means having 1 feature
            X, y, _ = self.__gen_vector_instances(corpus, max(next(corpus.edges()).features.keys()) + 1)
            return (X, y)


    @staticmethod
    def _are_vector_instances_already_computed(corpus):
        try:
            return next(corpus.edges()).features_vector is not None
        except StopIteration:
            # No edges at all in the corpus --> the non-existing feature_vectors are "already computed" (empty)
            return True


    @staticmethod
    def _convert_edges_features_reusing_computed_vector_instances(corpus):
        num_instances = sum(1 for _ in corpus.edges())
        try:
            num_features = next(corpus.edges()).features_vector.shape[1]
        except StopIteration:
            # No edges at all in the corpus --> no features
            num_features = 0

        X = scipy.sparse.lil_matrix((num_instances, num_features), dtype=np.float64)
        y = np.zeros(num_instances, order='C')  # -- see: http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC

        for edge_index, edge in enumerate(corpus.edges()):
            X[edge_index, :] = edge.features_vector
            y[edge_index] = edge.real_target

        X = X.tocsr()

        return (X, y)
