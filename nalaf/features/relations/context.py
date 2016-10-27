from nalaf.features.relations import EdgeFeatureGenerator


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
