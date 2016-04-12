from nalaf.features import FeatureGenerator
from gensim.models import Word2Vec
from nalaf import print_verbose
from spacy.en import English
import re


class WordEmbeddingsFeatureGenerator(FeatureGenerator):
    """
    DOCSTRING
    """

    def __init__(self, model_file, additive=0, multiplicative=1):
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
            wrd = re.sub('\d', '0', token.word.lower())
            if wrd in self.model:
                for index, value in enumerate(self.model[wrd]):
                    # value.item() since value is a numpy float and we want native python floats
                    token.features[feature_names[index]] = (self.additive + value.item()) * self.multiplicative


class DiscreteWordEmbeddingsFeatureGenerator(FeatureGenerator):
    """
    DOCSTRING
    """

    def __init__(self, model_file, n_bins=300):
        import numpy as np
        self.model = Word2Vec.load(model_file)

        data = np.vstack(self.model[word] for word in self.model.vocab)
        hist, self.bin_edges = np.histogram(data.flatten(), bins=n_bins)

        print_verbose('word embddings loaded with vocab size:', len(self.model.vocab))

    def generate(self, dataset):
        import numpy as np
        """
        :type dataset: nalaf.structures.data.Dataset
        """
        # create the feature names only once
        feature_names = ['embedding_{}'.format(index)
                         for index in range(self.model[next(iter(self.model.vocab))].shape[0])]
        for token in dataset.tokens():
            word = re.sub('\d', '0', token.word.lower())
            if word in self.model:
                digits = np.digitize(self.model[word], self.bin_edges)
                for index, value in enumerate(digits):
                    token.features[feature_names[index]] = str(value)


class BinarizedWordEmbeddingsFeatureGenerator(FeatureGenerator):
    """
    DOCSTRING
    """

    def __init__(self, model_file):
        import numpy as np
        self.model = Word2Vec.load(model_file)

        data = np.vstack(self.model[word] for word in self.model.vocab)
        self.pos_means = np.average(data, axis=0, weights=(data > 0))
        self.neg_means = np.average(data, axis=0, weights=(data < 0))

        print_verbose('word embddings loaded with vocab size:', len(self.model.vocab))

    def generate(self, dataset):
        import numpy as np
        """
        :type dataset: nalaf.structures.data.Dataset
        """
        # create the feature names only once
        feature_names = ['embedding_{}'.format(index)
                         for index in range(self.model[next(iter(self.model.vocab))].shape[0])]
        for token in dataset.tokens():
            word = re.sub('\d', '0', token.word.lower())
            if word in self.model:
                vector = self.model[word]
                binarized = np.where(vector > self.pos_means, '+', np.where(vector < self.neg_means, '-', '0'))
                for index, value in enumerate(binarized):
                    token.features[feature_names[index]] = value


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
                    token.features['brown'] = assignment[:i + 1]


class SpacyWordEmbeddingsFeatureGenerator(FeatureGenerator):
    def __init__(self, additive=0, multiplicative=1):
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
                                token.features[feature_names[index]] = (
                                                                           self.additive + value.item()) * self.multiplicative

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
