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

    def __init__(self, model_path=None, classification_threshold=0.0, use_tree_kernel=False, preprocess=True, **svc_parameters):
        assert not use_tree_kernel, NotImplementedError

        self.model_path = model_path if model_path is not None else tempfile.NamedTemporaryFile().name
        """the model (path) to read from / write to"""
        print_debug("SVM-Light model file path: " + self.model_path)

        self.classification_threshold = classification_threshold
        self.preprocess = preprocess

        self.svc_parameters = svc_parameters
        self.model = svm.SVC(**svc_parameters)

        self.global_feature_set = None
        self.allowed_features_names = None
        """
        Set of allowed feature names, if given
        """
        self.allowed_features_indexes = None
        """
        Set of allowed feature names -- If None, the allowed indexes are exactly all indexes of the training data
        """
        self.final_allowed_feature_mapping = None
        """
        A map (dictionary) of allowed (feature indexes) to (final index of the feature in the features vectors).

        The maximum value of the final index is equal or less than the maximum value of the original feature index
        and only equal in case all feature indexes are allowed. The final indexes are enumerated from 0 to n-1.

        That is, for example, say the corpus has features indexes = {0, 1, 2, 3, 4}, yet the allowed feature indexes
        are only = {1, 3, 4}. The resulting final_allowed_feature_mapping is = {1: 0, 3: 1, 4: 2}.
        """


    def train(self, training_corpus, global_feature_set):
        self.global_feature_set = global_feature_set  # The total/global feature set is actually not used, so far
        self.allowed_features_indexes, self.final_allowed_feature_mapping = \
            __class__._gen_allowed_and_final_mapping_features_keys(training_corpus)

        X, y = __class__._convert_edges_features_to_vector_instances(training_corpus, self.preprocess, self.final_allowed_feature_mapping)
        print_debug("Train SVC with #samples {} - #features {} - params: {}".format(X.shape[0], X.shape[1], str(self.model.get_params())))
        start = time.time()
        self.model.fit(X, y)
        end = time.time()
        print_debug("SVC train, running time: ", (end - start))
        return self


    def annotate(self, corpus):
        X, y = __class__._convert_edges_features_to_vector_instances(corpus, self.preprocess, self.final_allowed_feature_mapping)
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
    def _gen_vector_instances(corpus, global_feature_set, preprocess):

        allowed = [0, 1, 4, 5, 9, 11, 12, 13, 14, 15, 17, 18, 19, 22, 24, 26, 27, 30, 31, 32, 34, 38, 39, 40, 41, 45, 46, 49, 50, 51, 54, 56, 62, 65, 66, 71, 73, 77, 79, 80, 86, 87, 89, 90, 97, 98, 101, 104, 105, 108, 110, 112, 115, 117, 119, 120, 121, 122, 125, 126, 132, 136, 137, 141, 142, 143, 145, 148, 149, 151, 152, 153, 155, 157, 159, 160, 161, 162, 163, 164, 166, 168, 169, 170, 171, 172, 178, 182, 183, 185, 193, 196, 199, 208, 210, 211, 212, 213, 219, 225, 229, 230, 232, 233, 235, 236, 237, 238, 240, 241, 242, 244, 246, 248, 249, 254, 260, 261, 264, 265, 268, 276, 277, 283, 284, 286, 290, 294, 295, 298, 299, 301, 305, 311, 313, 319, 321, 326, 337, 338, 339, 340, 341, 343, 344, 345, 349, 354, 369, 370, 371, 377, 385, 386, 388, 389, 391, 392, 393, 395, 396, 405, 406, 407, 409, 410, 411, 412, 413, 415, 417, 418, 430, 432, 433, 434, 435, 440, 441, 444, 445, 446, 450, 451, 453, 460, 462, 463, 464, 465, 467, 468, 470, 471, 472, 473, 474, 475, 477, 479, 480, 482, 483, 484, 485, 486, 487, 497, 502, 504, 505, 506, 507, 508, 514, 515, 516, 517, 518, 519, 523, 527, 528, 532, 534, 544, 552, 553, 554, 555, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 570, 573, 578, 594, 596, 598, 599, 606, 607, 611, 617, 625, 630, 634, 635, 645, 647, 653, 654, 657, 658, 663, 665, 666, 667, 668, 669, 670, 671, 672, 673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686, 688, 695, 700, 705, 714, 716, 734, 735, 754, 756, 769, 770, 772, 774, 775, 776, 782, 785, 786, 795, 801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811, 812, 813, 814, 833, 840, 842, 843, 844, 845, 854, 858, 859, 861, 867, 874, 885, 888, 889, 891, 899, 908, 917, 920, 923, 935, 938, 971, 976, 977, 978, 979, 980, 981, 982, 996, 1002, 1003, 1006, 1007, 1008, 1014, 1015, 1016, 1024, 1028, 1035, 1036, 1046, 1048, 1051, 1055, 1056, 1057, 1058, 1059, 1063, 1064, 1065, 1066, 1067, 1070, 1071, 1075, 1078, 1079, 1080, 1081, 1083, 1084, 1085, 1086, 1087, 1088, 1089, 1090, 1100, 1104, 1111, 1125, 1127, 1128, 1131, 1140, 1141, 1143, 1153, 1160, 1166, 1168, 1170, 1171, 1172, 1175, 1178, 1180, 1181, 1182, 1183, 1186, 1187, 1189, 1190, 1191, 1192, 1194, 1195, 1197, 1200, 1201, 1202, 1203, 1204, 1205, 1206, 1207, 1208, 1209, 1210, 1211, 1214, 1215, 1219, 1220, 1221, 1223, 1224, 1225, 1226, 1227, 1228, 1229, 1230, 1231, 1232, 1234, 1239, 1241, 1242, 1243, 1244, 1245, 1261, 1262, 1264, 1265, 1266, 1267, 1268, 1273, 1278, 1285, 1288, 1289, 1302, 1303, 1306, 1307, 1308, 1312, 1313, 1316, 1317, 1318, 1319, 1320, 1321, 1322, 1326, 1327, 1328, 1330, 1331, 1332, 1334, 1338, 1341, 1342, 1344, 1345, 1346, 1347, 1348, 1349, 1350, 1351, 1352, 1353, 1361, 1363, 1366, 1369, 1375, 1376, 1377, 1379, 1380, 1381, 1382, 1383, 1384, 1398, 1399, 1400, 1401, 1402, 1406, 1407, 1408, 1409, 1410, 1411, 1412, 1413, 1414, 1421, 1422, 1423, 1424, 1425, 1426, 1427, 1428, 1429, 1431, 1432, 1436, 1437, 1438, 1464, 1465, 1466, 1467, 1470, 1472, 1476, 1477, 1478, 1479, 1480, 1481, 1482, 1483, 1485, 1488, 1490, 1492, 1495, 1498, 1499, 1500, 1503, 1504, 1505, 1506, 1507, 1508, 1509, 1510]

        def fun(X, y, corpus):
            for edge_index, edge in enumerate(corpus.edges()):
                for f_key in edge.features.keys():
                    if f_key in allowed:
                        f_index = f_key
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
    def _convert_edges_features_anew(corpus, preprocess, final_allowed_feature_mapping):

        if final_allowed_feature_mapping is None:
            _, final_allowed_feature_mapping = __class__._gen_allowed_and_final_mapping_features_keys(corpus)

        def fun(X, y, corpus):
            for edge_index, edge in enumerate(corpus.edges()):
                for f_key in edge.features.keys():
                    if f_key in final_allowed_feature_mapping:
                        f_index = final_allowed_feature_mapping[f_key]
                        value = edge.features[f_key]
                        X[edge_index, f_index] = value

                y[edge_index] = edge.real_target

            return X, y


        return __class__._create_instances(
            num_features=len(final_allowed_feature_mapping),
            corpus=corpus,
            preprocess=preprocess,
            setting_function=fun
        )


    @staticmethod
    def _convert_edges_features_to_vector_instances(corpus, preprocess, final_allowed_feature_mapping=None):
        if __class__._vector_instances_already_computed(corpus):
            return __class__._convert_edges_features_reusing_computed_vector_instances(corpus)
        else:
            return __class__._convert_edges_features_anew(corpus, preprocess, final_allowed_feature_mapping)


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
