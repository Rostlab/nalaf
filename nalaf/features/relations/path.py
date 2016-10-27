from nalaf.features.relations import EdgeFeatureGenerator
from relna.features.relations import TokenFeatureGenerator
from nltk.stem import PorterStemmer
from nalaf.utils.graph import get_path, build_walks


class PathFeatureGenerator(EdgeFeatureGenerator):
    """
    The length of the path from entity 1 to entity 2 and token features for the
    two tokens at the terminal of the path
    """
    def __init__(self, graphs):
        self.graphs = graphs
        """a dictionary of graphs to avoid recomputation of path"""
        self.stemmer = PorterStemmer()
        """an instance of PorterStemmer"""
        self.token_feature_generator = TokenFeatureGenerator()
        """an instance of TokenFeatureGenerator"""


    def generate(self, dataset, feature_set, is_training_mode):
        for edge in dataset.edges():
            head1 = edge.entity1.head_token
            head2 = edge.entity2.head_token
            sentence = edge.part.sentences[edge.sentence_id]
            path = []
            path = get_path(head1, head2, edge.part, edge.sentence_id, self.graphs)
            if len(path)==0:
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
        feature_name_1 = '45_len_tokens_' + str(len(path)) + '_[0]'
        feature_name_2 = '46_len_[0]'
        self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name_1)
        self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name_2, len(path))


    def path_constituents(self, path, edge, words, feature_set, is_training_mode):
        for token in path:
            if self.stemmer.stem(token.word) in words:
                feature_name_1 = '47_word_in_path_' + self.stemmer.stem(token.word) + '_[0]'
                self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name_1)


    def path_dependency_features(self, path, edge, feature_set, is_training_mode):
        for i in range(len(path)-1):
            token1 = path[i]
            token2 = path[i+1]
            for dep in token1.features['dependency_to']:
                if dep[0]==token2:
                    feature_name = '48_dep_'+dep[1]+'_forward_[0]'
                    self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)

            for dep in token2.features['dependency_to']:
                if dep[0]==token1:
                    feature_name = '49_dep_'+dep[1]+'_reverse_[0]'
                    self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)

        for i in range(1, len(path)-1):
            token = path[i]
            feature_name_1 = '50_internal_pos_' + token.features['pos'] + '_[0]'
            feature_name_2 = '51_internal_masked_txt_' + token.masked_text(edge.part) + '_[0]'
            feature_name_3 = '52_internal_txt_' + token.word + '_[0]'
            feature_name_4 = '53_internal_stem_' + self.stemmer.stem(token.word) + '_[0]'
            self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name_1)
            self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name_2)
            self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name_3)
            self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name_4)

        for i in range(2, len(path)-1):
            token1 = path[i]
            token2 = path[i+1]
            for dep in token1.features['dependency_to']:
                if dep[0]==token2:
                    feature_name = '54_internal_dep_'+dep[1]+'_forward_[0]'
                    self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)

            for dep in token2.features['dependency_to']:
                if dep[0]==token1:
                    feature_name = '55_internal_dep_'+dep[1]+'_reverse_[0]'
                    self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)


    def build_walk_paths(self, path, edge, feature_set, is_training_mode):
        internal_types = ''
        for token in path:
            ann_types = self.token_feature_generator.annotated_types(token, edge)
            for ann in ann_types:
                internal_types += '_'+ann
            internal_types += '_'
            feature_name = '56_token_path'+internal_types+'_[0]'
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
            for j in range(len(path)-1):
                current_walk = all_walks[i]
                if current_walk[j][0].features['dependency_from'][0]==path[i]:
                    dir_grams += 'F'
                else:
                    dir_grams += 'R'
                if i>=n-1:
                    style_gram = ''
                    style_gram = dir_grams[i-n+1:i+1]
                    edge_gram = 'dep_gram_' + style_gram

                    for k in range(1, n):
                        token = edge.part.sentences[edge.sentence_id][(path[i-(n-1)+k]).features['id']-1]
                        self.token_feature_generator.token_features(token, 'tok_'+style_gram, edge, feature_set, is_training_mode)

                    for k in range(n):
                        dep = current_walk[i-(n-1)+k][1]
                        feature_name = '57_dep_'+style_gram+'_'+str(k)+'_'+dep+'_[0]'
                        self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)
                        edge_gram += '_' + dep

                    feature_name = '58_'+edge_gram+'_[0]'
                    self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)

                    for ann1 in token1_anns:
                        for ann2 in token2_anns:
                            feature_name = '59_'+ann1+'_'+edge_gram+'_'+ann2+'_[0]'
                            self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)

                feature_name = '60_edge_directions_' + dir_grams + '_[0]'
                self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)


    def path_edge_features(self, path, edge, feature_set, is_training_mode):
        head1 = edge.entity1.head_token
        head2 = edge.entity2.head_token
        dependency_list = []
        for i in range(len(path)-1):
            token1 = path[i]
            token2 = path[i+1]
            dependency_list.append(token2.features['dependency_from'])
            dependency_list.append(token1.features['dependency_from'])

        for dependency in dependency_list:
            feature_name = '61_dep_'+dependency[1]+'_[0]'
            self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)
            feature_name = '62_txt_'+dependency[0].masked_text(edge.part)+'_[0]'
            self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)
            feature_name = '63_pos_'+dependency[0].features['pos']+'_[0]'
            self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)

            token1 = dependency[0]
            ann_types_1 = self.token_feature_generator.annotated_types(token1, edge)
            for ann in ann_types_1:
                feature_name = '64_ann_type_'+ann+'_[0]'
                self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)

            g_text = dependency[0].masked_text(edge.part)
            g_pos = dependency[0].features['pos']
            g_at = 'no_ann_type'

            for dep in dependency[0].features['dependency_to']:
                feature_name = '65_'+dep[1]+'_[0]'
                self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)
                feature_name = '66_txt_'+dep[0].masked_text(edge.part)+'_[0]'
                self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)
                feature_name = '67_pos_'+dep[0].features['pos']+'_[0]'
                self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)

                token2 = dep[0]
                ann_types_2 = self.token_feature_generator.annotated_types(token2, edge)
                for ann in ann_types_2:
                    feature_name = '68_ann_type_'+ann+'_[0]'
                    self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)

                d_text = token2.masked_text(edge.part)
                d_pos = token2.features['pos']
                d_at = 'no_ann_type'

                feature_name = '69_gov_'+g_text+'_'+d_text+'_[0]'
                self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)

                feature_name = '70_gov_'+g_pos+'_'+d_pos+'_[0]'
                self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)

                for ann1 in ann_types_1:
                    for ann2 in ann_types_2:
                        feature_name = '71_gov_'+ann1+'_'+ann2+'_[0]'
                        self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)

                for ann1 in ann_types_1:
                    feature_name = '72_triple_'+ann1+'_'+dependency[1]+'_'+d_at+'_[0]'
                    self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)
