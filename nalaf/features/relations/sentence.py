from nalaf.features.relations import EdgeFeatureGenerator


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
