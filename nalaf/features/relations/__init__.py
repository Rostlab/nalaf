import abc
from nalaf.features import FeatureGenerator
import re
from nalaf import print_debug, print_verbose


class EdgeFeatureGenerator(FeatureGenerator):
    """
    Abstract class for generating features for each edge in the dataset.
    Subclasses that inherit this class should:
    * Be named [Name]FeatureGenerator
    * Implement the abstract method generate
    * Append new items to the dictionary field "features" of each Edge in the dataset
    """

    @abc.abstractmethod
    def generate(self, dataset, feature_set, use_gold=True, use_pred=False):
        """
        :type dataset: nalaf.structures.data.Dataset
        """
        pass


    def add_to_feature_set(self, feature_set, edge, feature_name, value=1):
        """
        Return True if feature was added to feature_set. False, otherwise

        If the feature_name is None, the feature is not added in anycase. See: self.mk_feature_name
        """
        if feature_name is None:
            return False

        else:
            feature_name = self.__set_final_name(feature_name)

            if not feature_set.is_locked:
                feature_index = feature_set.get(feature_name, None)

                if feature_index is None:
                    feature_index = len(feature_set)
                    feature_set[feature_name] = feature_index
                    print_verbose("Feature map: {} == {} -- _1st_ value: {}".format(str(feature_index), feature_name, str(value)))

                edge.features[feature_index] = value
                return True

            else:
                feature_index = feature_set.get(feature_name, None)

                if feature_index is not None:
                    edge.features[feature_index] = value
                    return True
                else:
                    return False

    def __set_final_name(self, feature_name):
        if not re.search('\[-?[0-9]+\]$', feature_name):
            # Identify the window position --> TODO likely deletable from a edge feature generator
            feature_name = feature_name + "_[0]"

        if not feature_name.startswith(self.__class__.__name__):
            feature_name = self.__class__.__name__ + "::" + feature_name

        return feature_name


    def mk_feature_name(self, prefix, *args):
        if prefix is None:
            return None
        else:
            l = [str(x) for x in ([prefix] + list(args))]
            return "_".join(l)


    def gen_prefix_feat_name(self, field_prefix_feature, *args):
        prefix = self.__getattribute__(field_prefix_feature)
        pure_name = field_prefix_feature[field_prefix_feature.find("_") + 1:]  # Remove "prefix_"
        feature_name = self.mk_feature_name(prefix, pure_name, *args)
        # print_debug(feature_name, field_prefix_feature, args)
        return feature_name

    def add(self, feature_set, edge, field_prefix_feature, *args):
        feature_name = self.gen_prefix_feat_name(field_prefix_feature, *args)
        self.add_to_feature_set(feature_set, edge, feature_name)


    def add_with_value(self, feature_set, edge, field_prefix_feature, value, *args):
        feature_name = self.gen_prefix_feat_name(field_prefix_feature, *args)
        self.add_to_feature_set(feature_set, edge, feature_name, value=value)


from nalaf.features.relations import EdgeFeatureGenerator
from nltk.stem import PorterStemmer
from math import log2
from operator import itemgetter


class TokenFeatureGenerator(EdgeFeatureGenerator):
    """
    Token based features for each entity belonging to an edge
    """

    def __init__(
        self,
        prefix_txt=None,  # 73 in relna
        prefix_pos=None,  # 74
        prefix_masked_txt=None,  # 75
        prefix_stem_masked_txt=None,  # 76
        prefix_ann_type=None,  # 77
    ):
        self.stemmer = PorterStemmer()
        """an instance of the PorterStemmer()"""

        self.prefix_txt = prefix_txt
        self.prefix_pos = prefix_pos
        self.prefix_masked_txt = prefix_masked_txt
        self.prefix_stem_masked_txt = prefix_stem_masked_txt
        self.prefix_ann_type = prefix_ann_type


    @abc.abstractmethod
    def generate(self, dataset, feature_set, is_training_mode):
        """
        Does nothing directly
        """
        pass


    def token_features(self, token, addendum, edge, feature_set, is_training_mode):
        feature_name_1 = self.gen_prefix_feat_name("prefix_txt", addendum, token.word)
        self.add_to_feature_set(feature_set, edge, feature_name_1)

        feature_name_2 = self.gen_prefix_feat_name("prefix_pos", addendum, token.features['pos'])
        self.add_to_feature_set(feature_set, edge, feature_name_2)

        feature_name_3 = self.gen_prefix_feat_name("prefix_masked_txt", addendum, token.masked_text(edge.same_part))
        self.add_to_feature_set(feature_set, edge, feature_name_3)

        # TODO why stem of masked text? -- makes little sense -- See TODO in original loctext too
        feature_name_4 = self.gen_prefix_feat_name("prefix_stem_masked_txt", addendum, self.stemmer.stem(token.masked_text(edge.same_part)))
        self.add_to_feature_set(feature_set, edge, feature_name_4)

        ann_types = self.annotated_types(token, edge)
        for ann in ann_types:
            feature_name_5 = self.gen_prefix_feat_name("prefix_ann_type", addendum, ann)
            self.add_to_feature_set(feature_set, edge, feature_name_5)


    def annotated_types(self, token, edge):
        head1 = edge.entity1.head_token
        head2 = edge.entity2.head_token

        if not token.is_entity_part(edge.same_part):
            feature_name = 'no_ann_type'
            return [feature_name]
        else:
            ann_types = []
            if token.is_entity_part(edge.same_part):
                entity = token.get_entity(edge.same_part)
                feature_name_1 = entity.class_id
                ann_types.append(feature_name_1)
                if entity == edge.entity1:
                    feature_name_2 = 'entity1_' + edge.entity1.class_id
                    ann_types.append(feature_name_2)
                    return ann_types
                elif entity == edge.entity2:
                    feature_name_2 = 'entity2_' + edge.entity2.class_id
                    ann_types.append(feature_name_2)
                    return ann_types
                return ann_types


def calculateInformationGain(feature_set, dataset, output_file):
    number_pos_instances = 0
    number_neg_instances = 0

    for edge in dataset.edges():
        if edge.real_target == +1:
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
            if edge.real_target == +1:
                if value in edge.features.keys():
                    feature_present_in_pos += 1
                    total_feature_present += 1
                else:
                    feature_absent_in_pos += 1
                    total_feature_absent +=1
            if edge.real_target == -1:
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
