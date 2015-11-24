from nala.features import FeatureGenerator
from nltk.stem import PorterStemmer
import abc

class EdgeFeatureGenerator(FeatureGenerator):
    """
    Abstract class for generating features for each edge in the dataset.
    Subclasses that inherit this class should:
    * Be named [Name]FeatureGenerator
    * Implement the abstract method generate
    * Append new items to the dictionary field "features" of each Edge in the dataset
    """

    @abc.abstractmethod
    def generate(self, dataset):
        """
        :type dataset: nala.structures.data.Dataset
        """
        return

    def add_to_feature_set(self, edge, feature_name, value=1):
        if self.training_mode:
            if feature_name not in self.feature_set.keys():
                self.feature_set[feature_name] = len(self.feature_set.keys()) + 1
            edge.features[self.feature_set[feature_name]] = value
        else:
            if feature_name in self.feature_set.keys():
                edge.features[self.feature_set[feature_name]] = value


class TokenFeatureGenerator:
    """
    Token based features for each entity belonging to an edge
    """
    def __init__(self, feature_set, training_mode=True):
        self.feature_set = feature_set
        """the feature set for the dataset"""
        self.training_mode = training_mode
        """whether the mode is training or testing"""
        self.stemmer = PorterStemmer()
        """an instance of the PorterStemmer()"""

    def token_features(self, token, prefix, edge):
        feature_name_1 = '73_'+prefix+'txt_'+token.word+'_[0]'
        self.add_to_feature_set(edge, feature_name_1)
        feature_name_2 = '74_'+prefix+'pos_'+token.features['pos']+'_[0]'
        self.add_to_feature_set(edge, feature_name_2)
        feature_name_3 = '75_'+prefix+'txt_'+token.masked_text(edge.part)+'_[0]'
        self.add_to_feature_set(edge, feature_name_3)
        feature_name_4 = '76_'+prefix+'stem_'+self.stemmer.stem(token.masked_text(edge.part))+'_[0]'
        self.add_to_feature_set(edge, feature_name_4)
        ann_types = self.annotated_types(token, edge)
        for ann in ann_types:
            feature_name_5 = '77_'+prefix+'ann_type_'+ann+'_[0]'
            self.add_to_feature_set(edge, feature_name_5)

    def annotated_types(self, token, edge):
        head1 = edge.entity1.head_token
        head2 = edge.entity2.head_token
        if not token.is_entity_part(edge.part):
            feature_name = 'no_ann_type'
            return [feature_name]
        else:
            ann_types = []
            if token.is_entity_part(edge.part):
                entity = token.get_entity(edge.part)
                feature_name_1 = entity.class_id
                ann_types.append(feature_name_1)
                if entity==edge.entity1:
                    feature_name_2 = 'entity1_'+edge.entity1.class_id
                    ann_types.append(feature_name_2)
                    return ann_types
                elif entity==edge.entity2:
                    feature_name_2 = 'entity2_'+edge.entity2.class_id
                    ann_types.append(feature_name_2)
                    return ann_types
                return ann_types

    def add_to_feature_set(self, edge, feature_name, value=1):
        if self.training_mode:
            if feature_name not in self.feature_set.keys():
                self.feature_set[feature_name] = len(self.feature_set.keys()) + 1
            edge.features[self.feature_set[feature_name]] = value
        else:
            if feature_name in self.feature_set.keys():
                edge.features[self.feature_set[feature_name]] = value


def calculateInformationGain(feature_set, dataset, output_file):
    number_pos_instances = 0
    number_neg_instances = 0

    for edge in dataset.edges():
        if edge.target == 1:
            number_pos_instances += 1
        else:
            number_neg_instances += 1

    number_total_instances = number_pos_instances + number_neg_instances
    percentage_pos_instances = number_pos_instances / number_total_instances
    percentage_neg_instances = number_neg_instances / number_total_instances

    first_ent_component = -1 * (percentage_pos_instances * log2(percentage_pos_instances) + percentage_neg_instances * log2(percentage_neg_instances))
    feature_list = []
    for key, value in feature_set.items():
        feature_present_in_pos = 0
        feature_present_in_neg = 0
        feature_absent_in_pos = 0
        feature_absent_in_neg = 0
        total_feature_present = 0
        total_feature_absent = 0

        for edge in dataset.edges():
            if edge.target == 1:
                if value in edge.features.keys():
                    feature_present_in_pos += 1
                    total_feature_present += 1
                else:
                    feature_absent_in_pos += 1
                    total_feature_absent +=1
            if edge.target == -1:
                if value in edge.features.keys():
                    feature_present_in_neg += 1
                    total_feature_present += 1
                else:
                    feature_absent_in_neg += 1
                    total_feature_absent += 1

        percentage_pos_given_feature = 0
        percentage_neg_given_feature = 0
        if (total_feature_present > 0):
            percentage_pos_given_feature = feature_present_in_pos / total_feature_present
            percentage_neg_given_feature = feature_present_in_neg / total_feature_present

        percentage_pos_given_feature_log = 0
        percentage_neg_given_feature_log = 0
        if percentage_pos_given_feature > 0:
            percentage_pos_given_feature_log = log2(percentage_pos_given_feature)
        if percentage_neg_given_feature > 0:
            percentage_neg_given_feature_log = log2(percentage_neg_given_feature)

        second_emp_component_factor = percentage_pos_given_feature * percentage_pos_given_feature_log + \
                            percentage_neg_given_feature * percentage_neg_given_feature_log

        percentage_feature_given_pos = feature_present_in_pos / number_pos_instances
        percentage_feature_given_neg = feature_present_in_pos / number_neg_instances
        percentage_feature = percentage_feature_given_pos * percentage_pos_instances + \
                    percentage_feature_given_neg * percentage_neg_instances

        second_ent_component = percentage_feature * second_emp_component_factor
        percentage_pos_given_feature_component = 0
        percentage_neg_given_feature_component = 0
        if total_feature_absent>0:
            percentage_pos_given_feature_component = feature_absent_in_pos / total_feature_absent
            percentage_neg_given_feature_component = feature_absent_in_neg / total_feature_absent

        percentage_pos_given_feature_component_log = 0
        percentage_neg_given_feature_component_log = 0
        if percentage_pos_given_feature_component>0:
            percentage_pos_given_feature_component_log = log2(percentage_pos_given_feature_component)
        if percentage_neg_given_feature_component>0:
            percentage_neg_given_feature_component_log = log2(percentage_neg_given_feature_component)

        third_component_multi_factor = percentage_pos_given_feature_component * percentage_pos_given_feature_component_log + \
                percentage_neg_given_feature_component * percentage_neg_given_feature_component_log

        percentage_feature_comp_given_pos = feature_absent_in_pos / number_pos_instances
        percentage_feature_comp_given_neg = feature_absent_in_neg / number_neg_instances
        percentage_feature_comp = percentage_feature_comp_given_pos * percentage_pos_instances + \
                    percentage_feature_comp_given_neg * percentage_neg_instances

        third_ent_component = percentage_feature_comp * third_component_multi_factor
        entropy = first_ent_component + second_ent_component + third_ent_component

        feature_list.append([key, value, entropy])

    feature_list = sorted(feature_list, key=itemgetter(2), reverse=True)

    return feature_list
