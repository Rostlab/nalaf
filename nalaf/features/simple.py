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


class ExternalPredictedLabelsFeatureGenerator(FeatureGenerator):
    """
    For each token generates a simple float features:
    [SYSTEM NAME]-[LABEL]:[PROBABILITY] where:
        * [SYSTEM NAME] is the name of some other tagging system that was used to predict labels
        * [LABEL] is the label assigned by that system
        * [PROBABILITY] is the confidence assigned by that system

    The labels should be provided in an input file where:
        * for each token there is a new line with [LABEL]\t[PROBABILITY]
        * sequences are separated by an empty line
    """

    def __init__(self, system_name, input_file, weight=1):
        self.weight = weight
        """
        the weight of the external features, by default 1
        """
        self.input_file = input_file
        self.system_name = system_name

    def generate(self, dataset):
        """
        :type dataset: nalaf.structures.data.Dataset
        """

        with open(self.input_file) as file:
            for sentence in dataset.sentences():
                for token in sentence:
                    label, probability = file.readline().split('\t')
                    token.features['{}-{}'.format(self.system_name, label)] = self.weight * float(probability)

                file.readline()  # skip the empty line signifying new sentence
