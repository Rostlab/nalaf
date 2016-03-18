from nalaf.features import FeatureGenerator
from gensim.models import Word2Vec
from nalaf import print_verbose
from spacy.en import English


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
        # create the feature names only once
        feature_names = ['embedding_{}'.format(index)
                         for index in range(self.model[next(iter(self.model.vocab))].shape[0])]
        for token in dataset.tokens():
            if token.word.lower() in self.model:
                for index, value in enumerate(self.model[token.word.lower()]):
                    # value.item() since value is a numpy float and we want native python floats
                    token.features[feature_names[index]] = (self.additive + value.item()) * self.multiplicative

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

class SpacyWordEmbeddingsFeatureGenerator(FeatureGenerator):
    def __init__(self, additive=0,  multiplicative=1):
        self.additive = additive
        self.multiplicative = multiplicative
        self.nlp = English(parser=False, tagger=False, entity=False)

    def generate(self, dataset):
        """
        :type dataset: nalaf.structures.data.Dataset
        """
        # create the feature names only once
        feature_names = ['embedding_{}'.format(index) for index in range(self.nlp('the')[0].vector.shape[0])]
        for part in dataset.parts():
            spc = self.nlp(part.text)
            for sentence in part.sentences:
                for token in sentence:
                    for spacy_token in spc:
                        start = spacy_token.idx
                        end = start + len(spacy_token)
                        if start <= token.start < token.end <= end:
                            for index, value in enumerate(spacy_token.vector):
                                token.features[feature_names[index]] = (self.additive + value.item()) * self.multiplicative

                            break

class SpacyBrownClusteringFeatureGenerator(FeatureGenerator):
    def __init__(self):
        self.nlp = English(parser=False, tagger=False, entity=False)

    def generate(self, dataset):
        """
        :type dataset: nalaf.structures.data.Dataset
        """
        for part in dataset.parts():
            spc = self.nlp(part.text)
            for sentence in part.sentences:
                for token in sentence:
                    for spacy_token in spc:
                        start = spacy_token.idx
                        end = start + len(spacy_token)
                        if start <= token.start < token.end <= end:
                            token.features['brown_cluster'] = str(spacy_token.cluster)
                            break