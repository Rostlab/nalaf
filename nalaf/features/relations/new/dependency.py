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
from nalaf.utils.graphs import compute_shortest_path, compute_shortest_paths, path


class DependencyFeatureGenerator(EdgeFeatureGenerator):
    """
    Generate a combination of dependency-based features as roughly defined in Master's Thesis pages 40-42.

    Some feature additions, removals, or detail changes are also considered.

    """

    # TODO do kinda constituency parsing http://www.clips.ua.ac.be/pages/mbsp-tags

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
        f_OW_N_gram_lemma=None,
        f_OW_N_gram_pos=None,
        f_OW_N_gram_tokens_count=None,
        f_OW_N_gram_tokens_count_without_punct=None,
        #
        f_IW_N_gram_lemma=None,
        f_IW_N_gram_pos=None,
        f_IW_N_gram_tokens_count=None,
        f_IW_N_gram_tokens_count_without_punct=None,
        #
        f_LD_N_gram_lemma=None,
        f_LD_N_gram_pos=None,
        f_LD_N_gram_tokens_count=None,
        f_LD_N_gram_tokens_count_without_punct=None,
        #
        f_PD_N_gram_lemma=None,
        f_PD_N_gram_pos=None,
        f_PD_N_gram_tokens_count=None,
        f_PD_N_gram_tokens_count_without_punct=None,
    ):

        # Hyper parameters
        self.h_ow_size = h_ow_size
        self.h_ow_grams = h_ow_grams
        self.h_iw_size = h_iw_size
        self.h_iw_grams = h_iw_grams
        self.h_ld_grams = h_ld_grams
        self.h_pd_grams = h_pd_grams
        # Feature keys/names
        self.f_OW_N_gram_lemma = f_OW_N_gram_lemma
        self.f_OW_N_gram_pos = f_OW_N_gram_pos
        self.f_OW_N_gram_tokens_count = f_OW_N_gram_tokens_count
        self.f_OW_N_gram_tokens_count_without_punct = f_OW_N_gram_tokens_count_without_punct
        #
        self.f_IW_N_gram_lemma = f_IW_N_gram_lemma
        self.f_IW_N_gram_pos = f_IW_N_gram_pos
        self.f_IW_N_gram_tokens_count = f_IW_N_gram_tokens_count
        self.f_IW_N_gram_tokens_count_without_punct = f_IW_N_gram_tokens_count_without_punct
        #
        self.f_LD_N_gram_lemma = f_LD_N_gram_lemma
        self.f_LD_N_gram_pos = f_LD_N_gram_pos
        self.f_LD_N_gram_tokens_count = f_LD_N_gram_tokens_count
        self.f_LD_N_gram_tokens_count_without_punct = f_LD_N_gram_tokens_count_without_punct
        #
        self.f_PD_N_gram_lemma = f_PD_N_gram_lemma
        self.f_PD_N_gram_pos = f_PD_N_gram_pos
        self.f_PD_N_gram_tokens_count = f_PD_N_gram_tokens_count
        self.f_PD_N_gram_tokens_count_without_punct = f_PD_N_gram_tokens_count_without_punct


    def f(self, f_key, dependency_XX, ngram_N=None):
        """Return the final real name of the feature"""
        dependency_XX = dependency_XX[:2]
        return f_key.replace('XX', dependency_XX)


    def add_all(self, f_set, is_train, edge, tokens, dep_type, n_gram):
        tokens_n_grams = zip(*(tokens[i:] for i in range(0, n_gram)))

        s = (lambda string: '[' + string + ']')
        f = (lambda tokens_group, ft_key: s(' '.join(t.features[ft_key] for t in tokens_group)))

        for tokens_group in tokens_n_grams:
            lemmas = f(tokens_group, 'lemma')
            poses = f(tokens_group, 'pos')

            self.add(f_set, is_train, edge, self.f('f_XX_N_gram_lemma', dep_type), dep_type, n_gram, lemmas)
            self.add(f_set, is_train, edge, self.f('f_XX_N_gram_pos', dep_type), dep_type, n_gram, poses)

        count = len(tokens)
        count_without_punct = len(list(filter(lambda t: not t.features['is_punct'], tokens)))
        self.add_with_value(f_set, is_train, edge, self.f('f_XX_N_gram_tokens_count', dep_type), count, dep_type, n_gram)
        self.add_with_value(f_set, is_train, edge, self.f('f_XX_N_gram_tokens_count_without_punct', dep_type), count_without_punct, dep_type, n_gram)


    def generate(self, corpus, f_set, is_train):
        for edge in corpus.edges():
            sentence = edge.get_combined_sentence()

            # Remember, the edge's entities are sorted by offset
            ow1 = list(reversed(edge.entity1.prev_tokens(sentence, n=self.h_ow_size)))
            iw1 = edge.entity1.next_tokens(sentence, n=self.h_iw_size)
            ow2 = list(reversed(edge.entity2.prev_tokens(sentence, n=self.h_ow_size)))
            iw2 = edge.entity2.next_tokens(sentence, n=self.h_iw_size)

            _e1_next_token = edge.entity1.next_tokens(sentence, n=1)[0].features['id']
            _e2_head_token = edge.entity2.tokens[0].features['id']
            ld = sentence[(_e1_next_token):(_e2_head_token)]

            pd = compute_shortest_path(sentence, edge.entity1.head_token, edge.entity2.head_token)

            #

            for n_gram in self.h_ow_grams:
                self.add_all(f_set, is_train, edge, ow1, 'OW1', n_gram)

            for n_gram in self.h_iw_grams:
                self.add_all(f_set, is_train, edge, iw1, 'IW1', n_gram)

            for n_gram in self.h_ow_grams:
                self.add_all(f_set, is_train, edge, ow2, 'OW2', n_gram)

            for n_gram in self.h_iw_grams:
                self.add_all(f_set, is_train, edge, iw2, 'IW2', n_gram)

            for n_gram in self.h_ld_grams:
                self.add_all(f_set, is_train, edge, ld, 'LD', n_gram)

            for n_gram in self.h_pd_grams:
                self.add_all(f_set, is_train, edge, pd.tokens, 'PD', n_gram)
