"""
Sentence-based features implementation as succintly described in Shrikant's Master's Thesis (Section 4.5):
https://github.com/juanmirocks/LocText-old-ShrikantThesis/files/474428/MasterThesis.pdf
"""

from nalaf.features.relations import EdgeFeatureGenerator
from nalaf.features.util import masked_text
from nalaf.structures.data import Part
from collections import Counter


class SentenceFeatureGenerator(EdgeFeatureGenerator):
    """
    General generator to extract features out of the instances' sentences.

    Few document-based features are also generated. These will likely be moved soon to another class.
    """

    def __init__(
        self,

        f_counts_individual,
        f_counts_total,

        # Actually, LINEAR-DEPENDENCY BASED
        f_counts_in_between_individual,
        f_counts_in_between_total,

        f_order,

        f_bow,
        f_pos,

        f_tokens_count,

        # Likely, to be deleted
        f_tokens_count_before,
        f_tokens_count_after,

        f_sentence_is_negated,
        f_main_verbs,

        # The following are actually DOCUMENT-BASED features

        f_entity1_count,
        f_entity2_count,
        f_diff_sents_together_count,
    ):

        self.f_counts_individual = f_counts_individual
        self.f_counts_total = f_counts_total
        self.f_counts_in_between_individual = f_counts_in_between_individual
        self.f_counts_in_between_total = f_counts_in_between_total

        self.f_order = f_order

        self.f_bow = f_bow
        self.f_pos = f_pos

        self.f_tokens_count = f_tokens_count
        self.f_tokens_count_before = f_tokens_count_before
        self.f_tokens_count_after = f_tokens_count_after

        self.f_sentence_is_negated = f_sentence_is_negated
        self.f_main_verbs = f_main_verbs

        self.f_entity1_count = f_entity1_count
        self.f_entity2_count = f_entity2_count
        self.f_diff_sents_together_count = f_diff_sents_together_count


    def generate(self, corpus, f_set, use_gold, use_pred):
        assert not (use_gold and use_pred), "No support for both"

        self.extract_abbreviation_synonyms(corpus, use_gold, use_pred)

        for docid, document in corpus.documents.items():
            for edge in document.edges():

                sentence = edge.get_combined_sentence()

                entities_in_sentences = edge.get_any_entities_in_sentences(predicted=use_pred)
                total_count = 0
                # We sort to have a deterministic order creation of the features
                for e_class_id in sorted(entities_in_sentences):
                    entities = entities_in_sentences[e_class_id]
                    # TODO this is wrong for other entitiey types nor appearing in the edge
                    # TODO also what about if the same entity type appears in both ends of the same edge? as in a protein-protein relation --> Just rest the counts of the edge
                    individual_count = len(entities) - 1  # rest 1, as one is already one of the edge's entities --
                    assert individual_count >= 0
                    total_count += individual_count
                    self.add_with_value(f_set, edge, 'f_counts_individual', individual_count, 'int', 'individual', e_class_id)

                self.add_with_value(f_set, edge, 'f_counts_total', total_count, 'int', 'total (all classes)')

                entities_between_entities = edge.get_any_entities_between_entities(predicted=use_pred)
                total_count = 0
                # We sort to have a deterministic order creation of the features
                for e_class_id in sorted(entities_between_entities):
                    entities = entities_between_entities[e_class_id]
                    individual_count = len(entities)
                    total_count += individual_count
                    self.add_with_value(f_set, edge, 'f_counts_in_between_individual', individual_count, 'int', 'individual', e_class_id)

                self.add_with_value(f_set, edge, 'f_counts_in_between_total', total_count, 'int', 'total (all classes)')

                order = edge.entity1.class_id < edge.entity2.class_id
                if order:
                    self.add(f_set, edge, 'f_order')

                for token in sentence:
                    self.add(f_set, edge, 'f_bow', masked_text(token, edge.same_part, use_gold, use_pred, token_map=lambda t: t.features['lemma'], token_is_number_fun=lambda _: "NUM"))
                    self.add(f_set, edge, 'f_pos', token.features['coarsed_pos'])

                self.add_with_value(f_set, edge, 'f_tokens_count', len(sentence))

                # Remember, the edge's entities are sorted, i.e. e1.offset < e2.offset
                _e1_first_token_index = edge.entity1.tokens[0].features['tmp_id']
                _e2_last_token_index = edge.entity2.tokens[-1].features['tmp_id']
                assert _e1_first_token_index < _e2_last_token_index, (docid, sentence, edge.entity1.text, edge.entity2.text, _e1_first_token_index, _e2_last_token_index)

                self.add_with_value(f_set, edge, 'f_tokens_count_before', len(sentence[:_e1_first_token_index]))
                self.add_with_value(f_set, edge, 'f_tokens_count_after', len(sentence[(_e2_last_token_index+1):]))

                #

                if Part.is_negated(sentence):
                    self.add(f_set, edge, "f_sentence_is_negated")

                #

                verbs = set(Part.get_main_verbs(sentence, token_map=lambda t: t.features["lemma"]))

                if len(verbs) == 0:
                    self.add(f_set, edge, "f_main_verbs", "NO_MAIN_VERB")
                else:
                    for v in verbs:
                        self.add(f_set, edge, "f_main_verbs", v)

                counters = {}
                for part in document:
                    for entity in (part.annotations if use_gold else part.predicted_annotations):
                        ent_type_counter = counters.get(entity.class_id, Counter())
                        ent_key = __class__.entity2key(entity)
                        ent_type_counter.update([ent_key])
                        counters[entity.class_id] = ent_type_counter

                e1_key = __class__.entity2key(edge.entity1)
                e1_count = counters[edge.entity1.class_id][e1_key]
                self.add_with_value(f_set, edge, 'f_entity1_count', e1_count)

                e2_key = __class__.entity2key(edge.entity2)
                e2_count = counters[edge.entity2.class_id][e2_key]
                self.add_with_value(f_set, edge, 'f_entity2_count', e2_count)

                together_counter = Counter()
                diff_sentences = {}
                for aux_edge in document.edges():
                    if aux_edge.e1_sentence_id == aux_edge.e2_sentence_id:
                        together_key = __class__.edge2key(aux_edge)

                        sents = diff_sentences.get(together_key, [])
                        if aux_edge.e1_sentence_id not in sents:
                            sents.append(aux_edge.e1_sentence_id)
                            diff_sentences[together_key] = sents
                            together_counter.update([together_key])

                together_key = __class__.edge2key(edge)
                together_count = together_counter[together_key]
                if together_count > 0:
                    self.add_with_value(f_set, edge, 'f_diff_sents_together_count', together_count)


    @staticmethod
    def entity2key(entity):
        ent_norms = list(entity.norms.values())
        if len(ent_norms) > 0 and ent_norms[0] is not None:
            return ent_norms[0]
        else:
            return entity.text.lower()


    @staticmethod
    def edge2key(edge):
        e1_key = __class__.entity2key(edge.entity1)
        e2_key = __class__.entity2key(edge.entity2)

        if edge.entity1.class_id < edge.entity2.class_id:
            return e1_key + "|" + e2_key
        else:
            return e2_key + "|" + e1_key


    def extract_abbreviation_synonyms(self, corpus, use_gold, use_pred):
        """
        Apply simple heuristic to know if some entities are abbreviations of another one
        The protein x is abbreviation of protein y if they are written as: y (x)"
        In the end, more generically, we call it a "synonym" relationship
        """
        assert not (use_gold and use_pred), "No support for both"
        entities = corpus.entities() if use_gold else corpus.predicted_entities()

        for entity in entities:
            prev2 = entity.prev_tokens(entity.sentence, 2)
            next1 = entity.next_tokens(entity.sentence, 1)
            in_parenthesis = len(prev2) == 2 and prev2[-1].word == "(" and len(next1) == 1 and next1[0].word == ")"

            if (in_parenthesis):
                prev_entity = prev2[0].get_entity(entity.part, use_gold, use_pred)

                if prev_entity is not None and prev_entity.class_id == entity.class_id:
                    # We could combine features already -- Yet, give more freedom to final clients to use the synonym's features or not
                    # merged_binary_features = {key: (b1 or b2) for ((key, b1), (_, b2)) in zip(prev_entity.features.items(), entity.features.items())}

                    prev_entity.features['synonym'] = entity
                    entity.features['synonym'] = prev_entity
