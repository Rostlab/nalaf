from nalaf.features.relations import EdgeFeatureGenerator
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


class NamedEntityCountFeatureGenerator(EdgeFeatureGenerator):
    """
    Generates Named Entity Count for each sentence that contains an edge
    """

    def __init__(self, entity_type, prefix):
        self.entity_type = entity_type
        """type of entity"""
        self.prefix = prefix


    def generate(self, dataset, feature_set, is_training_mode):
        for edge in dataset.edges():
            entities = edge.part.get_entities_in_sentence(edge.sentence_id, self.entity_type)
            num_entities = len(entities)
            feature_name = self.mk_feature_name(self.prefix, self.entity_type, 'count', num_entities)
            self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name, value=1)


class BagOfWordsFeatureGenerator(EdgeFeatureGenerator):
    """
    Generates Bag of Words representation for each sentence that contains an edge

    :type feature_set: nalaf.structures.data.FeatureDictionary
    :type stop_words: list[str]
    :type is_training_mode: bool
    """
    def __init__(self, stop_words=None):
        if stop_words is None:
            stop_words = stopwords.words('english')
        self.stop_words = stop_words
        """a list of stop words"""


    def generate(self, dataset, feature_set, is_training_mode):
        for edge in dataset.edges():
            sentence = edge.part.sentences[edge.sentence_id]
            bow_map = {}
            for token in sentence:
                if token.word not in self.stop_words and not token.features['is_punct']:
                    feature_name = '2_bow_text_' + token.word + '_[0]'
                    self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)
                    if token.is_entity_part(edge.part):
                        bow_string = 'ne_bow_' + token.word + '_[0]'
                        if bow_string not in bow_map.keys():
                            bow_map[bow_string] = 0
                        bow_map[bow_string] = bow_map[bow_string]+1
            for key, value in bow_map.items():
                feature_name = '3_'+key
                self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name, value)


class StemmedBagOfWordsFeatureGenerator(EdgeFeatureGenerator):
    """
    Generates stemmed Bag of Words representation for each sentence that contains
    an edge, using the function given in the argument.

    By default it uses Porter stemmer

    :type feature_set: nalaf.structures.data.FeatureDictionary
    :type stemmer: nltk.stem.PorterStemmer
    :type stop_words: list[str]
    :type is_training_mode: bool
    """

    def __init__(self, stop_words=[]):
        self.stemmer = PorterStemmer()
        """an instance of the PorterStemmer"""
        self.stop_words = stop_words
        """a list of stop words"""


    def generate(self, dataset, feature_set, is_training_mode):
        for edge in dataset.edges():
            sentence = edge.part.sentences[edge.sentence_id]

            if is_training_mode:
                for token in sentence:
                    if self.stemmer.stem(token.word) not in self.stop_words and not token.features['is_punct']:
                        feature_name = '4_bow_stem_' + self.stemmer.stem(token.word) + '_[0]'
                        self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)
