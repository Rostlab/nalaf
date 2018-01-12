from nalaf.features.relations import EdgeFeatureGenerator
from nltk.stem import PorterStemmer


class LinearDistanceFeatureGenerator(EdgeFeatureGenerator):
    """
    The absolute distance between the two entities in the edge.
    If distance is greater than 5 (default), add to feature set.
    Also add the actual distance between the two entities.

    :param feature_set: the feature set for the dataset
    :type feature_set: nalaf.structures.data.FeatureDictionary
    :param distance: the number of tokens between the two entities, default 5
    :type distance: int
    :param training_mode: indicates whether the mode is training or testing
    :type training_mode: bool
    """

    def __init__(
        self, distance=5,
        prefix_entity_linear_distance_greater_than=None,
        prefix_entity_linear_distance_lesser_than=None,
        prefix_entity_linear_distance=None
    ):
        self.distance = distance
        self.prefix_entity_linear_distance_greater_than = prefix_entity_linear_distance_greater_than
        self.prefix_entity_linear_distance_lesser_than = prefix_entity_linear_distance_lesser_than
        self.prefix_entity_linear_distance = prefix_entity_linear_distance

    def generate(self, dataset, feature_set, is_training_mode):
        for edge in dataset.edges():
            entity1_number = edge.entity1.head_token.features['id']
            entity2_number = edge.entity2.head_token.features['id']
            distance = abs(entity1_number - entity2_number)
            if distance > self.distance:
                feature_name = self.gen_prefix_feat_name("prefix_entity_linear_distance_greater_than", 5)
                self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)
            else:
                feature_name = self.gen_prefix_feat_name("prefix_entity_linear_distance_lesser_than", 5)
                self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)

            feature_name = self.gen_prefix_feat_name("prefix_entity_linear_distance")
            self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name, value=distance)


class EntityOrderFeatureGenerator(EdgeFeatureGenerator):
    """
    The is the order of the entities in the sentence.  Whether entity1 occurs
    first or entity2 occurs first.

    :param feature_set: the feature set for the dataset
    :type feature_set: nalaf.structures.data.FeatureDictionary
    :param training_mode: indicates whether the mode is training or testing
    :type training_mode: bool
    """

    def __init__(
        self,
        prefix_order_entity1_entity2,
        prefix_order_entity2_entity1,
    ):
        self.prefix_order_entity1_entity2 = prefix_order_entity1_entity2
        self.prefix_order_entity2_entity1 = prefix_order_entity2_entity1


    def generate(self, dataset, feature_set, is_training_mode):
        for edge in dataset.edges():
            if edge.entity1.offset < edge.entity2.offset:
                feature_name = self.gen_prefix_feat_name("prefix_order_entity1_entity2")
            else:
                feature_name = self.gen_prefix_feat_name("prefix_order_entity2_entity1")

            self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)


class IntermediateTokensFeatureGenerator(EdgeFeatureGenerator):
    """
    Generate the bag of words representation, masked text, stemmed text and
    parts of speech tag for each of the tokens present between two entities in
    an edge.
    """

    def __init__(
        self,

        prefix_fwd_bow_intermediate=None,
        prefix_fwd_bow_intermediate_masked=None,
        prefix_fwd_stem_intermediate=None,
        prefix_fwd_pos_intermediate=None,

        prefix_bkd_bow_intermediate=None,
        prefix_bkd_bow_intermediate_masked=None,
        prefix_bkd_stem_intermediate=None,
        prefix_bkd_pos_intermediate=None,

        prefix_bow_intermediate=None,
        prefix_bow_intermediate_masked=None,
        prefix_stem_intermediate=None,
        prefix_pos_intermediate=None,
    ):
        self.stemmer = PorterStemmer()
        """an instance of PorterStemmer"""

        self.prefix_fwd_bow_intermediate = prefix_fwd_bow_intermediate
        self.prefix_fwd_bow_intermediate_masked = prefix_fwd_bow_intermediate_masked
        self.prefix_fwd_stem_intermediate = prefix_fwd_stem_intermediate
        self.prefix_fwd_pos_intermediate = prefix_fwd_pos_intermediate

        self.prefix_bkd_bow_intermediate = prefix_bkd_bow_intermediate
        self.prefix_bkd_bow_intermediate_masked = prefix_bkd_bow_intermediate_masked
        self.prefix_bkd_stem_intermediate = prefix_bkd_stem_intermediate
        self.prefix_bkd_pos_intermediate = prefix_bkd_pos_intermediate

        self.prefix_bow_intermediate = prefix_bow_intermediate
        self.prefix_bow_intermediate_masked = prefix_bow_intermediate_masked
        self.prefix_stem_intermediate = prefix_stem_intermediate
        self.prefix_pos_intermediate = prefix_pos_intermediate


    def generate(self, dataset, feature_set, is_training_mode):
        for edge in dataset.edges():
            sentence = edge.same_part.sentences[edge.same_sentence_id]

            if edge.entity1.head_token.features['id'] < edge.entity2.head_token.features['id']:
                first = edge.entity1.head_token.features['id']
                second = edge.entity2.head_token.features['id']

                for i in range(first + 1, second):
                    token = sentence[i]

                    feature_name = self.gen_prefix_feat_name('prefix_fwd_bow_intermediate', token.word)
                    self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)
                    feature_name = self.gen_prefix_feat_name('prefix_fwd_bow_intermediate_masked', token.masked_text(edge.same_part))
                    self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)
                    feature_name = self.gen_prefix_feat_name('prefix_fwd_stem_intermediate', self.stemmer.stem(token.word))
                    self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)
                    feature_name = self.gen_prefix_feat_name('prefix_fwd_pos_intermediate', token.features['pos'])
                    self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)

            else:
                first = edge.entity2.head_token.features['id']
                second = edge.entity1.head_token.features['id']

                for i in range(first + 1, second):
                    token = sentence[i]

                    feature_name = self.gen_prefix_feat_name('prefix_bkd_bow_intermediate', token.word)
                    self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)
                    feature_name = self.gen_prefix_feat_name('prefix_bkd_bow_intermediate_masked', token.masked_text(edge.same_part))
                    self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)
                    feature_name = self.gen_prefix_feat_name('prefix_bkd_stem_intermediate', self.stemmer.stem(token.word))
                    self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)
                    feature_name = self.gen_prefix_feat_name('prefix_bkd_pos_intermediate', token.features['pos'])
                    self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)

            for i in range(first + 1, second):
                token = sentence[i]

                feature_name = self.gen_prefix_feat_name('prefix_bow_intermediate', token.word)
                self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)
                feature_name = self.gen_prefix_feat_name('prefix_bow_intermediate_masked', token.masked_text(edge.same_part))
                self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)
                feature_name = self.gen_prefix_feat_name('prefix_stem_intermediate', self.stemmer.stem(token.word))
                self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)
                feature_name = self.gen_prefix_feat_name('prefix_pos_intermediate', token.features['pos'])
                self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)
