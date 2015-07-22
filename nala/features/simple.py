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
            token.features['word[0]'] = token.word
