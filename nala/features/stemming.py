from nala.features import FeatureGenerator
from nltk.stem import PorterStemmer


class PorterStemFeatureGenerator(FeatureGenerator):
    """
    Generates stem features based on the values of the tokens themselves.
        * stem[0] = the value of the word itself stemmed

    Uses the NLTK implementation of the Porter Stemmer. The original Porter Stemming
    algorithm can be found here http://tartarus.org/~martin/PorterStemmer/.

    Implements the abstract class FeatureGenerator.
    """

    def __init__(self):
        self.stemmer = PorterStemmer()

    def generate(self, dataset):
        """
        :type dataset: nala.structures.data.Dataset
        """
        for token in dataset.tokens():
            token.features['stem'] = self.stemmer.stem(token.word)
