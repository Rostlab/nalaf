from nala.features import FeatureGenerator
from gensim.models import Word2Vec


class WordEmbeddingsFeatureGenerator(FeatureGenerator):
    """
    DOCSTRING
    """

    def __init__(self, model_file):
        self.model = Word2Vec.load(model_file)

    def generate(self, dataset):
        """
        :type dataset: nala.structures.data.Dataset
        """
        for token in dataset.tokens():
            if token.word.lower() in self.model:
                for index, value in enumerate(self.model[token.word.lower()]):
                    # value.item() since value is a numpy float
                    # and we want native python floats
                    token.features['embedding_{}'.format(index)] = value.item()
