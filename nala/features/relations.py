from nala.features import FeatureGenerator
from nltk.stem import PorterStemmer

class NamedEntityCountFeatureGenerator(FeatureGenerator):
    """
    Generates Named Entity Count for each sentence that contains an edge

    :type entity_type: str
    :type mode: str
    :type feature_set: dict
    :type training_mode: bool
    """
    def __init__(self, entity_type, feature_set, training_mode=True):
        self.entity_type = entity_type
        """type of entity"""
        self.training_mode = training_mode
        """whether the mode is training or testing"""
        self.feature_set = feature_set
        """the feature set"""

    def generate(self, dataset):
        for edge in dataset.edges():
            entities = edge.part.get_entities_in_sentence(edge.sentence_id, self.entity_type)
            feature_name = self.entity_type + '_count_[' + str(len(entities)) + ']'
            if self.training_mode:
                if feature_name not in self.feature_set:
                    self.feature_set[feature_name] = len(self.feature_set.keys())
                edge.features[self.feature_set[feature_name]] = 1
            else:
                if feature_name in feature_set.keys():
                    edge.features[self.feature_set[feature_name]] = 1

class BagOfWordsFeatureGenerator(FeatureGenerator):
    """
    Generates Bag of Words representation for each sentence that contains an edge

    :type feature_set: nala.structures.data.FeatureDictionary
    :type training_mode: bool
    """
    def __init__(self, feature_set, training_mode=True):
        self.feature_set = feature_set
        """the feature set for the dataset"""
        self.training_mode = training_mode
        """whether the mode is training or testing"""

    def generate(self, dataset):
        for edge in dataset.edges():
            sentence = edge.part.sentences[edge.sentence_id]
            if self.training_mode:
                for token in sentence:
                    feature_name = 'bow_' + token.word + '[0]'
                    if feature_name not in self.feature_set:
                        self.feature_set[feature_name] = len(self.feature_set.keys())
                    edge.features[self.feature_set[feature_name]] = 1
            else:
                for token in sentence:
                    feature_name = 'bow_' + token.word + '[0]'
                    if feature_name in feature_set.keys():
                        edge.features[self.feature_set[feature_name]]

class StemmedBagOfWordsFeatureGenerator(FeatureGenerator):
    """
    Generates stemmed Bag of Words representation for each sentence that contains
    an edge, using the function given in the argument.

    By default it uses Porter stemmer

    :type feature_set: nala.structures.data.FeatureDictionary
    :type training_mode: bool
    :type stemmer: function
    """

    def __init__(self, feature_set, training_mode=True):

        self.feature_set = feature_set
        """the feature set for the dataset"""
        self.training_mode = training_mode
        """whether the mode is training or testing"""
        self.stemmer = PorterStemmer()

    def generate(self, dataset):
        for edge in dataset.edges():
            sentence = edge.part.sentences[edge.sentence_id]
            if self.training_mode:
                for token in sentence:
                    feature_name = 'bow_stem_' + self.stemmer.stem(token.word) + '[0]'
                    if feature_name not in self.feature_set:
                        self.feature_set[feature_name] = len(self.feature_set.keys())
                    edge.features[self.feature_set[feature_name]] = 1
            else:
                for token in sentence:
                    feature_name = 'bow_stem_' + self.stemmer.stem(token.word) + '[0]'
                    if feature_name in feature_set.keys():
                        edge.features[self.feature_set[feature_name]]


class 
