from nala.features import FeatureGenerator


class SimpleFeatureGenerator(FeatureGenerator):
    """
    Generates simple features based on the values of the tokens themselves.
        * word[0] = the value of the word itself

    Implements the abstract class FeatureGenerator.
    """

    def generate(self, dataset):
        """
        :type dataset: nala.structures.data.Dataset
        """
        for token in dataset.tokens():
            token.features['word'] = token.word


class SentenceMarkerFeatureGenerator(FeatureGenerator):
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
