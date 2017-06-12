from nalaf.features.relations import EdgeFeatureGenerator
from nalaf.features.relations import TokenFeatureGenerator
from nltk.stem import PorterStemmer
from nalaf.utils.graph import get_path, build_walks


class PathFeatureGenerator(EdgeFeatureGenerator):
    """
    The length of the path from entity 1 to entity 2 and token features for the
    two tokens at the terminal of the path
    """
    def __init__(
        self,

        graphs,

        token_feature_generator,

        prefix_45_len_tokens,
        prefix_46_len,
        prefix_47_word_in_path,
        prefix_48_dep_forward,
        prefix_49_dep_reverse,
        prefix_50_internal_pos,
        prefix_51_internal_masked_txt,
        prefix_52_internal_txt,
        prefix_53_internal_stem,
        prefix_54_internal_dep_forward,
        prefix_55_internal_dep_reverse,
        prefix_56_token_path,
        prefix_57_dep_style_gram,
        prefix_58_edge_gram,
        prefix_59_ann_edge_gram,
        prefix_60_edge_directions,
        prefix_61_dep_1,
        prefix_62_masked_txt_dep_0,
        prefix_63_pos_dep_0,
        prefix_64_ann_type_1,
        prefix_65_dep_to_1,
        prefix_66_masked_txt_dep_to_0,
        prefix_67_pos_to,
        prefix_68_ann_type_2,
        prefix_69_gov_g_text,
        prefix_70_gov_g_pos,
        prefix_71_gov_anns,
        prefix_72_triple,
    ):
        self.graphs = graphs
        """a dictionary of graphs to avoid recomputation of path"""
        self.stemmer = PorterStemmer()
        """an instance of PorterStemmer"""
        self.token_feature_generator = token_feature_generator

        self.prefix_45_len_tokens = prefix_45_len_tokens
        self.prefix_46_len = prefix_46_len
        self.prefix_47_word_in_path = prefix_47_word_in_path
        self.prefix_48_dep_forward = prefix_48_dep_forward
        self.prefix_49_dep_reverse = prefix_49_dep_reverse
        self.prefix_50_internal_pos = prefix_50_internal_pos
        self.prefix_51_internal_masked_txt = prefix_51_internal_masked_txt
        self.prefix_52_internal_txt = prefix_52_internal_txt
        self.prefix_53_internal_stem = prefix_53_internal_stem
        self.prefix_54_internal_dep_forward = prefix_54_internal_dep_forward
        self.prefix_55_internal_dep_reverse = prefix_55_internal_dep_reverse
        self.prefix_56_token_path = prefix_56_token_path
        self.prefix_57_dep_style_gram = prefix_57_dep_style_gram
        self.prefix_58_edge_gram = prefix_58_edge_gram
        self.prefix_59_ann_edge_gram = prefix_59_ann_edge_gram
        self.prefix_60_edge_directions = prefix_60_edge_directions
        self.prefix_61_dep_1 = prefix_61_dep_1
        self.prefix_62_masked_txt_dep_0 = prefix_62_masked_txt_dep_0
        self.prefix_63_pos_dep_0 = prefix_63_pos_dep_0
        self.prefix_64_ann_type_1 = prefix_64_ann_type_1
        self.prefix_65_dep_to_1 = prefix_65_dep_to_1
        self.prefix_66_masked_txt_dep_to_0 = prefix_66_masked_txt_dep_to_0
        self.prefix_67_pos_to = prefix_67_pos_to
        self.prefix_68_ann_type_2 = prefix_68_ann_type_2
        self.prefix_69_gov_g_text = prefix_69_gov_g_text
        self.prefix_70_gov_g_pos = prefix_70_gov_g_pos
        self.prefix_71_gov_anns = prefix_71_gov_anns
        self.prefix_72_triple = prefix_72_triple


    def generate(self, dataset, feature_set, is_training_mode):
        for edge in dataset.edges():
            head1 = edge.entity1.head_token
            head2 = edge.entity2.head_token
            sentence = edge.same_part.sentences[edge.same_sentence_id]
            path = []
            path = get_path(head1, head2, edge.same_part, edge.same_sentence_id, self.graphs)
            if len(path) == 0:
                path = [head1, head2]
            self.path_length_features(path, edge, feature_set, is_training_mode)
            self.token_feature_generator.token_features(path[0], 'token_term_1_', edge, feature_set, is_training_mode)
            self.token_feature_generator.token_features(path[-1], 'token_term_2_', edge, feature_set, is_training_mode)
            self.path_dependency_features(path, edge, feature_set, is_training_mode)
            base_words = ['interact', 'bind', 'coactivator', 'complex', 'mediate']
            words = []
            for word in base_words:
                words.append(self.stemmer.stem(word))
            self.path_constituents(path, edge, words, feature_set, is_training_mode)
            self.path_grams(2, path, edge, feature_set, is_training_mode)
            self.path_grams(3, path, edge, feature_set, is_training_mode)
            self.path_grams(4, path, edge, feature_set, is_training_mode)
            self.path_edge_features(path, edge, feature_set, is_training_mode)


    def path_length_features(self, path, edge, feature_set, is_training_mode):
        feature_name_1 = self.gen_prefix_feat_name('prefix_45_len_tokens', str(len(path)))
        self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name_1)

        feature_name_2 = self.gen_prefix_feat_name('prefix_46_len')
        self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name_2, value=len(path))


    def path_constituents(self, path, edge, words, feature_set, is_training_mode):
        for token in path:
            if self.stemmer.stem(token.word) in words:
                feature_name_1 = self.gen_prefix_feat_name('prefix_47_word_in_path', self.stemmer.stem(token.word))
                self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name_1)


    def path_dependency_features(self, path, edge, feature_set, is_training_mode):
        for i in range(len(path) - 1):
            token1 = path[i]
            token2 = path[i + 1]

            for dep in token1.features['dependency_to']:
                if dep[0] == token2:
                    feature_name = self.gen_prefix_feat_name('prefix_48_dep_forward', dep[1])
                    self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)

            for dep in token2.features['dependency_to']:
                if dep[0] == token1:
                    feature_name = self.gen_prefix_feat_name('prefix_49_dep_reverse', dep[1])
                    self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)

        for i in range(1, len(path) - 1):
            token = path[i]
            feature_name_1 = self.gen_prefix_feat_name('prefix_50_internal_pos', token.features['pos'])
            feature_name_2 = self.gen_prefix_feat_name('prefix_51_internal_masked_txt', token.masked_text(edge.same_part))
            feature_name_3 = self.gen_prefix_feat_name('prefix_52_internal_txt', token.word)
            feature_name_4 = self.gen_prefix_feat_name('prefix_53_internal_stem', self.stemmer.stem(token.word))
            self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name_1)
            self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name_2)
            self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name_3)
            self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name_4)

        for i in range(2, len(path) - 1):
            token1 = path[i]
            token2 = path[i + 1]
            for dep in token1.features['dependency_to']:
                if dep[0] == token2:
                    feature_name = self.gen_prefix_feat_name('prefix_54_internal_dep_forward', dep[1])
                    self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)

            for dep in token2.features['dependency_to']:
                if dep[0] == token1:
                    feature_name = self.gen_prefix_feat_name('prefix_55_internal_dep_reverse', dep[1])
                    self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)


    def build_walk_paths(self, path, edge, feature_set, is_training_mode):
        internal_types = ''
        for token in path:
            ann_types = self.token_feature_generator.annotated_types(token, edge)
            for ann in ann_types:
                internal_types += '_'+ann
            internal_types += '_'
            feature_name = self.gen_prefix_feat_name('prefix_56_token_path', internal_types)
            self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)


    def path_grams(self, n, path, edge, feature_set, is_training_mode):
        token1 = path[0]
        token2 = path[-1]
        token1_anns = self.token_feature_generator.annotated_types(token1, edge)
        token2_anns = self.token_feature_generator.annotated_types(token2, edge)
        self.build_walk_paths(path, edge, feature_set, is_training_mode)
        all_walks = build_walks(path)

        for i in range(len(all_walks)):
            dir_grams = ''

            for j in range(len(path) - 1):
                current_walk = all_walks[i]
                if current_walk[j][0].features['dependency_from'][0] == path[i]:
                    dir_grams += 'F'  # Forward
                else:
                    dir_grams += 'R'  # Reverse

                if i>=n-1:
                    style_gram = ''
                    style_gram = dir_grams[i-n+1:i + 1]
                    edge_gram = 'dep_gram_' + style_gram

                    for k in range(1, n):
                        token = edge.same_part.sentences[edge.same_sentence_id][(path[i-(n-1)+k]).features['id']-1]
                        self.token_feature_generator.token_features(token, 'tok_'+style_gram, edge, feature_set, is_training_mode)

                    for k in range(n):
                        dep = current_walk[i-(n-1)+k][1]
                        feature_name = self.gen_prefix_feat_name('prefix_57_dep_style_gram', style_gram, str(k), dep)
                        self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)
                        edge_gram += '_' + dep

                    feature_name = self.gen_prefix_feat_name('prefix_58_edge_gram', edge_gram)
                    self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)

                    for ann1 in token1_anns:
                        for ann2 in token2_anns:
                            feature_name = self.gen_prefix_feat_name('prefix_59_ann_edge_gram', ann1, edge_gram, ann2)
                            self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)

            # Note: relna code had this within the 2nd inner loop. This was different to what's in original LocText
            # and likely an unintended bug. The difference was spotted by Madhukar
            feature_name = self.gen_prefix_feat_name('prefix_60_edge_directions', dir_grams)
            self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)


    def path_edge_features(self, path, edge, feature_set, is_training_mode):
        head1 = edge.entity1.head_token
        head2 = edge.entity2.head_token

        dependency_list = []
        for i in range(len(path) - 1):
            token1 = path[i]
            token2 = path[i + 1]
            dependency_list.append(token2.features['dependency_from'])
            dependency_list.append(token1.features['dependency_from'])

        for dependency in dependency_list:
            feature_name = self.gen_prefix_feat_name('prefix_61_dep_1', dependency[1])
            self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)
            feature_name = self.gen_prefix_feat_name('prefix_62_masked_txt_dep_0', dependency[0].masked_text(edge.same_part))
            self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)
            feature_name = self.gen_prefix_feat_name('prefix_63_pos_dep_0', dependency[0].features['pos'])
            self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)

            token1 = dependency[0]
            ann_types_1 = self.token_feature_generator.annotated_types(token1, edge)
            for ann in ann_types_1:
                feature_name = self.gen_prefix_feat_name('prefix_64_ann_type_1', ann)
                self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)

            g_text = dependency[0].masked_text(edge.same_part)
            g_pos = dependency[0].features['pos']
            g_at = 'no_ann_type'


            # TODO Juanmi: I do not understand why the extra inner loop here (not in original LocText)
            # I think it's just trying to get the dep-to token. However, that should already be included in dependency[1]
            # to the best of my knowledge. Also `depdnency_to` is now broken
            for dep in dependency[0].features['dependency_to']:
                feature_name = self.gen_prefix_feat_name('prefix_65_dep_to_1', dep[1])
                self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)
                feature_name = self.gen_prefix_feat_name('prefix_66_masked_txt_dep_to_0', dep[0].masked_text(edge.same_part))
                self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)
                feature_name = self.gen_prefix_feat_name('prefix_67_pos_to', dep[0].features['pos'])
                self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)

                token2 = dep[0]
                ann_types_2 = self.token_feature_generator.annotated_types(token2, edge)
                for ann in ann_types_2:
                    feature_name = self.gen_prefix_feat_name('prefix_68_ann_type_2', ann)
                    self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)

                d_text = token2.masked_text(edge.same_part)
                d_pos = token2.features['pos']
                d_at = 'no_ann_type'

                feature_name = self.gen_prefix_feat_name('prefix_69_gov_g_text', g_text, d_text)
                self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)

                feature_name = self.gen_prefix_feat_name('prefix_70_gov_g_pos', g_pos, d_pos)
                self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)

                for ann1 in ann_types_1:
                    for ann2 in ann_types_2:
                        feature_name = self.gen_prefix_feat_name('prefix_71_gov_anns', ann1, ann2)
                        self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)

                for ann1 in ann_types_1:
                    feature_name = self.gen_prefix_feat_name('prefix_72_triple', ann1, dependency[1], d_at)
                    self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)
