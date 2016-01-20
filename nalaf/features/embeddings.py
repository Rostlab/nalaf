from nalaf.features import FeatureGenerator
from gensim.models import Word2Vec
from nalaf import print_verbose

class WordEmbeddingsFeatureGenerator(FeatureGenerator):
    """
    DOCSTRING
    """

    def __init__(self, model_file, additive=0,  multiplicative=1):
        self.model = Word2Vec.load(model_file)
        self.additive = additive
        self.multiplicative = multiplicative
        print_verbose('word embddings loaded with vocab size:', len(self.model.vocab))

    def generate(self, dataset):
        """
        :type dataset: nalaf.structures.data.Dataset
        """
        for token in dataset.tokens():
            if token.word.lower() in self.model:
                for index, value in enumerate(self.model[token.word.lower()]):
                    # value.item() since value is a numpy float
                    # and we want native python floats
                    token.features['embedding_{}'.format(index)] = (self.additive + value.item()) * self.multiplicative


class BrownClusteringFeatureGenerator(FeatureGenerator):
        """
        DOCSTRING
        """
        def __init__(self, model_file, weight=1):
            with open(model_file, encoding='utf-8') as file:
                self.clusters = {str(line.split()[1]): line.split()[0] for line in file.readlines()}
            self.weight = weight

        def generate(self, dataset):
            """
            :type dataset: nalaf.structures.data.Dataset
            """
            for token in dataset.tokens():
                if token.word in self.clusters:
                    assignment = self.clusters[token.word]
                    for i in range(len(assignment)):
                        token.features['brown'] = assignment[:i+1]
