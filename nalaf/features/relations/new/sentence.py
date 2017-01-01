"""
Sentence-based features implementation as succintly described in Shrikant's Master's Thesis (Section 4.5):
https://github.com/juanmirocks/LocText-old-ShrikantThesis/files/474428/MasterThesis.pdf
"""

from nalaf.features.relations import EdgeFeatureGenerator


class SentenceFeatureGenerator(EdgeFeatureGenerator):
    """
    Generate sentence-based features as roughly defined in Master's Thesis pages 40-42.

    Some feature additions, removals, or detail changes are also considered.

    It strictly gets features that are based on the sentence: ONLY
    """

    def __init__(
        self,
        f_counts,
        f_counts_in_between,
        f_order,
        f_bow,
        f_pos,
        f_tokens_count
    ):

        self.f_counts = f_counts
        self.f_counts_in_between = f_counts_in_between
        self.f_order = f_order
        self.f_bow = f_bow
        self.f_pos = f_pos
        self.f_tokens_count = f_tokens_count


    def generate(self, corpus, f_set, is_train):
        for edge in corpus.edges():
            sentence = edge.get_combined_sentence()

            for e_class_id, entities in edge.get_any_entities_in_sentences(predicted=False).items():
                count = len(entities)
                if e_class_id in [edge.entity1.class_id, edge.entity2.class_id]:
                    # Rest -1 thus removing each consider pair of entities
                    count -= 1

                # TODO feature select with integer value or binary representation ?
                # self.add(f_set, is_train, edge, 'f_counts', 'binary', e_class_id, 'is', count)
                self.add_with_value(f_set, is_train, edge, 'f_counts', count, 'int', e_class_id)

            for e_class_id, entities in edge.get_any_entities_between_entities(predicted=False).items():
                count = str(len(entities))
                # TODO feature select with integer value or binary representation ?
                # self.add(f_set, is_train, edge, 'f_counts_in_between', 'binary', e_class_id, 'is', count)
                self.add_with_value(f_set, is_train, edge, 'f_counts_in_between', count, 'int', e_class_id)

            order = edge.entity1.class_id < edge.entity2.class_id
            if order:
                self.add(f_set, is_train, edge, 'f_order')

            for token in sentence:
                self.add(f_set, is_train, edge, 'f_bow', token.features['lemma'])
                self.add(f_set, is_train, edge, 'f_pos', token.features['pos'])

            self.add_with_value(f_set, is_train, edge, 'f_tokens_count', len(sentence))
