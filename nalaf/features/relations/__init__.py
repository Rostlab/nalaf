import abc
from nalaf.features import FeatureGenerator
import re
from nalaf import print_debug, print_verbose


class EdgeFeatureGenerator(FeatureGenerator):
    """
    Abstract class for generating features for each edge in the dataset.
    Subclasses that inherit this class should:
    * Be named [Name]FeatureGenerator
    * Implement the abstract method generate
    * Append new items to the dictionary field "features" of each Edge in the dataset
    """

    @abc.abstractmethod
    def generate(self, dataset, feature_set, is_training_mode):
        """
        :type dataset: nalaf.structures.data.Dataset
        """
        pass


    def add_to_feature_set(self, feature_set, is_training_mode, edge, feature_name, value=1):
        """
        Return True if feature was added to feature_set. False, otherwise

        If the feature_name is None, the feature is not added in anycase. See: self.mk_feature_name
        """
        if feature_name is None:
            return False

        else:
            if not re.search('\[-?[0-9]+\]$', feature_name):
                feature_name = feature_name + "_[0]"  # See logic of definition in: FeatureDictionary

            if is_training_mode:
                if feature_name not in feature_set.keys():
                    index = len(feature_set.keys()) + 1
                    feature_set[feature_name] = index
                    print_verbose("Feature map: {} == {}".format(str(index), feature_name))
                edge.features[feature_set[feature_name]] = value
                return True
            else:
                if feature_name in feature_set.keys():
                    edge.features[feature_set[feature_name]] = value
                    return True
                else:
                    return False


    def mk_feature_name(self, prefix, *args):
        if prefix is None:
            return None
        else:
            l = [str(x) for x in ([prefix] + list(args))]
            return "_".join(l)
