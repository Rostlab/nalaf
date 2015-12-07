import re

from nalaf.features import FeatureGenerator


class SimpleFeatureGenerator(FeatureGenerator):
    """
    Generates simple features based on the values of the tokens themselves.
        * word[0] = the value of the word itself

    Implements the abstract class FeatureGenerator.
    """

    def generate(self, dataset):
        """
        :type dataset: nalaf.structures.data.Dataset
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
        :type dataset: nalaf.structures.data.Dataset
        """
        for sentence in dataset.sentences():
            sentence[0].features['BOS'] = 1
            sentence[-1].features['EOS'] = 1


class NonAsciiFeatureGenerator(FeatureGenerator):
    """
    Generates a simple binary features with shows
    whether the token contains non ascii characters or not.
    """

    def generate(self, dataset):
        """
        :type dataset: nalaf.structures.data.Dataset
        """
        for token in dataset.tokens():
            if re.search('[^\x00-\x7F]', token.word):
                token.features['non_ascii'] = 1
