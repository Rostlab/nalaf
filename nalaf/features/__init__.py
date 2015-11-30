import abc


class FeatureGenerator:
    """
    Abstract class for generating features for each token in the dataset.
    Subclasses that inherit this class should:
    * Be named [Name]FeatureGenerator
    * Implement the abstract method generate
    * Append new items to the dictionary field "features" of each Token in the dataset
    """
    @abc.abstractmethod
    def generate(self, dataset):
        """
        :type dataset: nalaf.structures.data.Dataset
        """
        return


def eval_binary_feature(feature_dict, feature_name, evaluator, *args):
    """
    Calls the provided callable with the provided arguments which evaluates to True or False.
    If the evaluation results in True add a new feature to the features dictionary with the provided feature name.

    :param feature_dict: the target feature dictionary where the feature should be added
    :param feature_name: the feature name to be used
    :param evaluator: any callable that evaluates to True or False
    :param args: arguments needed for the callable
    :type feature_dict: nalaf.structures.data.FeatureDictionary
    :type feature_name: str
    """
    if evaluator(*args):
        feature_dict[feature_name] = 1
