from nala.features import FeatureGenerator


class SimpleExtendedFeatureGenerator(FeatureGenerator):
    """
    Generates BOS and EOS features
        * BOS[0] = is it at the beginning of a sentence?
        * EOS[0] = is it at the end of a sentence?

    Implements the abstract class FeatureGenerator.
    """

    def generate(self, dataset):
        """
        :type dataset: nala.structures.data.Dataset
        """
        for sentence in dataset.sentences():
            sentence[0].features['BOS'] = 1
            sentence[-1].features['EOS'] = 1
