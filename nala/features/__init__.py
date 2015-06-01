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
        :type dataset: structures.data.Dataset
        """
        return
