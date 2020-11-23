"""
Combined dependency-based features implementation as succintly described in Shrikant's Master's Thesis (Section 4.5):
https://github.com/juanmirocks/LocText-old-ShrikantThesis/files/474428/MasterThesis.pdf

The implementation consider 4 types of dependency types:

* OW (1 and 2): Outer Window == tokens at the outer side of an entity (1 or 2)
* IW (1 and 2): Inner Window == tokens at the inner side of an entity (1 or 2)
* LD: Linear Dependency == tokens within the two entities
* PD: Parsing Dependency == real dependency parsing obtained from spaCy's library

"""

from nalaf.features.relations import EdgeFeatureGenerator
from nalaf.utils.graphs import compute_shortest_path, Path
from nalaf.features.util import masked_text
from nalaf.structures.data import Part


class DependencyFeatureGenerator(EdgeFeatureGenerator):
    """
    Generate a combination of dependency-based features as roughly defined in Master's Thesis pages 40-42.

    Some feature additions, removals, or detail changes are also considered.

    """

    def __init__(
        self,
        # Hyper parameters
        h_ow_size=4,  # outer window size
        h_ow_grams=[1, 2, 3, 4],
        h_iw_size=4,  # inner window size
        h_iw_grams=[1, 2, 3, 4],
        h_ld_grams=[1, 2, 3, 4],
        h_pd_grams=[1, 2, 3, 4],
        # Feature keys/names
        f_OW_bow_N_gram=None,
        f_OW_pos_N_gram=None,
        f_OW_tokens_count=None,
        f_OW_tokens_count_without_punct=None,
        f_OW_is_negated=None,
        #
        f_IW_bow_N_gram=None,
        f_IW_pos_N_gram=None,
        f_IW_tokens_count=None,
        f_IW_tokens_count_without_punct=None,
        f_IW_is_negated=None,
        #
        f_LD_bow_N_gram=None,
        f_LD_pos_N_gram=None,
        f_LD_tokens_count=None,
        f_LD_tokens_count_without_punct=None,
        f_LD_is_negated=None,
        #
        f_PD_bow_N_gram=None,
        f_PD_pos_N_gram=None,
        f_PD_tokens_count=None,
        f_PD_tokens_count_without_punct=None,
        f_PD_is_negated=None,
        # Specific ones for PD
        f_PD_undirected_edges_N_gram=None,
        f_PD_directed_edges_N_gram=None,
        f_PD_full_N_gram=None,
        # Extra features
        # None now
    ):

        # Hyper parameters
        self.h_ow_size = h_ow_size
        self.h_ow_grams = h_ow_grams
        self.h_iw_size = h_iw_size
        self.h_iw_grams = h_iw_grams
        self.h_ld_grams = h_ld_grams
        self.h_pd_grams = h_pd_grams
        # Feature keys/names
        self.f_OW_bow_N_gram = f_OW_bow_N_gram
        self.f_OW_pos_N_gram = f_OW_pos_N_gram
        self.f_OW_tokens_count = f_OW_tokens_count
        self.f_OW_tokens_count_without_punct = f_OW_tokens_count_without_punct
        self.f_OW_is_negated = f_OW_is_negated
        #
        self.f_IW_bow_N_gram = f_IW_bow_N_gram
        self.f_IW_pos_N_gram = f_IW_pos_N_gram
        self.f_IW_tokens_count = f_IW_tokens_count
        self.f_IW_tokens_count_without_punct = f_IW_tokens_count_without_punct
        self.f_IW_is_negated = f_IW_is_negated
        #
        self.f_LD_bow_N_gram = f_LD_bow_N_gram
        self.f_LD_pos_N_gram = f_LD_pos_N_gram
        self.f_LD_tokens_count = f_LD_tokens_count
        self.f_LD_tokens_count_without_punct = f_LD_tokens_count_without_punct
        self.f_LD_is_negated = f_LD_is_negated
        ####
        # Parsing Dependencies has more features
        ####
        # Regular ones
        self.f_PD_bow_N_gram = f_PD_bow_N_gram
        self.f_PD_pos_N_gram = f_PD_pos_N_gram
        self.f_PD_tokens_count = f_PD_tokens_count
        self.f_PD_tokens_count_without_punct = f_PD_tokens_count_without_punct
        self.f_PD_is_negated = f_PD_is_negated
        # Specific ones for PD
        self.f_PD_undirected_edges_N_gram = f_PD_undirected_edges_N_gram
        self.f_PD_directed_edges_N_gram = f_PD_directed_edges_N_gram
        self.f_PD_full_N_gram = f_PD_full_N_gram

        ####
        # Extra features
        ####
        # None now


    def generate(self, corpus, f_set, use_gold, use_pred):
        assert not (use_gold and use_pred), "No support for both"

        for docid, document in corpus.documents.items():
            for edge in document.edges():
                sentence = edge.get_combined_sentence()

                # Remember, the edge's entities are sorted, i.e. e1.offset < e2.offset
                _e1_last_token_index = edge.entity1.tokens[-1].features['tmp_id']
                _e2_first_token_index = edge.entity2.tokens[0].features['tmp_id']
                assert _e1_last_token_index < _e2_first_token_index, (docid, sentence, edge.entity1.text, edge.entity2.text, _e1_last_token_index, _e2_first_token_index)
                _e1_head_token_index = edge.entity1.head_token.features['tmp_id']
                _e2_head_token_index = edge.entity2.head_token.features['tmp_id']
                assert _e1_head_token_index < _e2_head_token_index, (docid, sentence, edge.entity1.text, edge.entity2.text, _e1_head_token_index, _e2_head_token_index)

                dependency_paths = [
                    #
                    # Commented out as of now: they do not seem to provide any performance benefit yet cost running time
                    #
                    # Path(
                    #     name='OW1',
                    #     tokens=edge.entity1.prev_tokens(sentence, n=self.h_ow_size, include_ent_first_token=True, mk_reversed=True),
                    #     is_edge_type_constant=True,
                    #     there_is_target=False,
                    #     default_n_grams=self.h_ow_grams,
                    # ),
                    # Path(
                    #     name='IW1',
                    #     tokens=edge.entity1.next_tokens(sentence, n=self.h_iw_size, include_ent_last_token=True),
                    #     is_edge_type_constant=True,
                    #     there_is_target=False,
                    #     default_n_grams=self.h_iw_grams,
                    # ),
                    #
                    # Path(
                    #     name='IW2',
                    #     tokens=edge.entity2.prev_tokens(sentence, n=self.h_iw_size, include_ent_first_token=True, mk_reversed=True),
                    #     is_edge_type_constant=True,
                    #     there_is_target=False,
                    #     default_n_grams=self.h_iw_grams,
                    # ),
                    # Path(
                    #     name='OW2',
                    #     tokens=edge.entity2.next_tokens(sentence, n=self.h_ow_size, include_ent_last_token=True),
                    #     is_edge_type_constant=True,
                    #     there_is_target=False,
                    #     default_n_grams=self.h_ow_grams,
                    # ),

                    Path(
                        name='LD',
                        tokens=sentence[_e1_last_token_index:_e2_first_token_index + 1],
                        is_edge_type_constant=True,
                        default_n_grams=self.h_ld_grams,
                    ),

                    compute_shortest_path(sentence, _e1_head_token_index, _e2_head_token_index).change_name('PD').change_default_n_grams(self.h_pd_grams)
                ]

                for dep_path in dependency_paths:
                    dep_type = dep_path.name

                    for n_gram in dep_path.default_n_grams:
                        self.add_n_grams(f_set, use_gold, use_pred, edge, dep_path, dep_type, n_gram)

                    count = len(dep_path.middle)
                    count_without_punct = len(list(filter(lambda node: not node.token.features['is_punct'], dep_path.middle)))
                    self.add_with_value(f_set, edge, self.f('f_XX_tokens_count', dep_type), count, dep_type)
                    self.add_with_value(f_set, edge, self.f('f_XX_tokens_count_without_punct', dep_type), count_without_punct, dep_type)

                    if Part.is_negated([node.token for node in dep_path.middle]):
                        self.add(f_set, edge, self.f('f_XX_is_negated', dep_type), dep_type)


    def f(self, feat_key, dependency_XX, ngram_N=None):
        """Return the final real name of the feature"""
        dependency_XX = dependency_XX[:2]
        return feat_key.replace('XX', dependency_XX)


    def add_n_grams(self, f_set, use_gold, use_pred, edge, path, dep_type, n_gram):

        def token_feat(tok_f_key):
            return (lambda t: t.features[tok_f_key])

        def add_groups(gen_f_key, path_str_fun, token_fun=None):
            groups = path_str_fun(n_gram, token_fun) if token_fun else path_str_fun(n_gram)

            for n_gram_group in groups:
                self.add(f_set, edge, self.f(gen_f_key, dep_type), dep_type, n_gram, n_gram_group)

        #
        # Regular features for all dependency paths types/names
        #

        add_groups('f_XX_bow_N_gram', path.strs_n_gram_token_only, lambda token: masked_text(token, edge.same_part, use_gold, use_pred, token_map=token_feat('lemma'), token_is_number_fun=lambda _: "NUM"))
        add_groups('f_XX_pos_N_gram', path.strs_n_gram_token_only, token_feat('coarsed_pos'))

        #
        #  Dedicated features for PD only
        #

        if not path.is_edge_type_constant:
            assert path.name == "PD"

            add_groups('f_PD_undirected_edges_N_gram', path.strs_n_gram_undirected_edge_only)
            add_groups('f_PD_directed_edges_N_gram', path.strs_n_gram_directed_edge_only)
            add_groups('f_PD_full_N_gram', path.strs_n_gram_full)
