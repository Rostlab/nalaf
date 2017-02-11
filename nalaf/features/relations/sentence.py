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


    def generate(self, dataset, feature_set, is_training_mode, use_gold, use_pred):

        for edge in dataset.edges():

            sentences_entities = []

            sentences_entities += edge.same_part.get_entities_in_sentence(edge.e1_sentence_id, self.entity_type)
            if (edge.e1_sentence_id != edge.e2_sentence_id):
                sentences_entities += edge.same_part.get_entities_in_sentence(edge.e2_sentence_id, self.entity_type)

            num_entities = len(sentences_entities)
            feature_name = self.mk_feature_name(self.prefix, self.entity_type, 'count', num_entities)
            # TODO we could set it as real value - ⚠️ that's what `entityhead::named_entity_count` did
            self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name, value=1)


class BagOfWordsFeatureGenerator(EdgeFeatureGenerator):
    """
    Generates Bag of Words representation for each sentence that contains an edge

    :type feature_set: nalaf.structures.data.FeatureDictionary
    :type stop_words: list[str]
    :type is_training_mode: bool
    """

    def __init__(
        self,
        stop_words=None,
        #
        prefix_bow_text=None,
        prefix_ne_bow_count=None,
    ):

        if stop_words is None:
            stop_words = stopwords.words('english')
        self.stop_words = stop_words

        self.prefix_bow_text = prefix_bow_text
        self.prefix_ne_bow_count = prefix_ne_bow_count


    def generate(self, dataset, feature_set, is_training_mode):
        for edge in dataset.edges():
            sentence = edge.same_part.sentences[edge.same_sentence_id]
            bow_map = {}
            for token in sentence:

                bow_string = token.word

                if bow_string not in self.stop_words and not token.features['is_punct']:
                    feature_name = self.gen_prefix_feat_name("prefix_bow_text", bow_string)
                    self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)

                    if token.is_entity_part(edge.same_part):
                        if bow_string not in bow_map.keys():
                            bow_map[bow_string] = 0
                        bow_map[bow_string] = bow_map[bow_string] + 1

            for ne_bow_key, count in bow_map.items():
                feature_name = self.gen_prefix_feat_name("prefix_ne_bow_count", ne_bow_key)
                self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name, value=count)


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

    def __init__(
        self, stop_words=[],
        prefix_bow_stem=None
    ):
        self.stemmer = PorterStemmer()
        """an instance of the PorterStemmer"""
        self.stop_words = stop_words

        self.prefix_bow_stem = prefix_bow_stem


    def generate(self, dataset, feature_set, is_training_mode):
        for edge in dataset.edges():
            sentence = edge.same_part.sentences[edge.same_sentence_id]

            if is_training_mode:
                for token in sentence:
                    if self.stemmer.stem(token.word) not in self.stop_words and not token.features['is_punct']:
                        feature_name = self.gen_prefix_feat_name("prefix_bow_stem", self.stemmer.stem(token.word))
                        self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)
