from nalaf.features.relations import EdgeFeatureGenerator


class LinearDistanceFeatureGenerator(EdgeFeatureGenerator):
    """
    The absolute distance between the two entities in the edge.
    If distance is greater than 5, add to feature set.
    Also add the actual distance between the two entities.

    :param feature_set: the feature set for the dataset
    :type feature_set: nalaf.structures.data.FeatureDictionary
    :param distance: the number of tokens between the two entities, default 5
    :type distance: int
    :param training_mode: indicates whether the mode is training or testing
    :type training_mode: bool
    """

    def __init__(self, distance=5):
        self.distance = distance
        """the distance parameter"""


    def generate(self, dataset, feature_set, is_training_mode):
        for edge in dataset.edges():
            entity1_number = edge.entity1.head_token.features['id']
            entity2_number = edge.entity2.head_token.features['id']
            distance = abs(entity1_number - entity2_number)
            if distance > self.distance:
                feature_name = '31_entity_linear_distance_greater_than_[5]'
                self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)
            else:
                feature_name = '31_entity_linear_distance_lesser_than_[5]'
                self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)
            feature_name = '32_entity_linear_distance_[0]'
            self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name, value=distance)
