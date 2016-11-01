import abc
from nalaf import print_debug
import time

_SPACY_NLP_ENGLISH_WITHOUT_PARSER = None
_SPACY_NLP_ENGLISH_WITH_PARSER = None


def _get_spacy_nlp_english(load_parser):
    import spacy

    start = time.time()
    print_debug("Spacy NLP English, Parser: {} -- INIT START".format(str(load_parser)))

    if load_parser is True:
        nlp = spacy.load('en', entity=False)
    else:
        nlp = spacy.load('en', parser=False, entity=False)

    print_debug("Spacy NLP English, Parser: {} -- INIT END   : ".format(str(load_parser)), (time.time() - start))

    return nlp


def get_spacy_nlp_english(load_parser=False):

    global _SPACY_NLP_ENGLISH_WITHOUT_PARSER
    global _SPACY_NLP_ENGLISH_WITH_PARSER

    if load_parser is True:
        if _SPACY_NLP_ENGLISH_WITH_PARSER is None:
            _SPACY_NLP_ENGLISH_WITH_PARSER = _get_spacy_nlp_english(load_parser)

        return _SPACY_NLP_ENGLISH_WITH_PARSER

    else:
        if _SPACY_NLP_ENGLISH_WITHOUT_PARSER is None:
            _SPACY_NLP_ENGLISH_WITHOUT_PARSER = _get_spacy_nlp_english(load_parser)

        return _SPACY_NLP_ENGLISH_WITHOUT_PARSER


class FeatureGenerator:
    """
    Abstract class for generating features for each token in the dataset.
    Subclasses that inherit this class should:
    * Be named [Name]FeatureGenerator
    * Implement the abstract method generate
    * Append new items to the dictionary field "features" of each Token in the dataset
    """

    @abc.abstractmethod
    def generate(self, dataset):
        """
        :type dataset: nalaf.structures.data.Dataset
        """
        return


def eval_binary_feature(feature_dict, feature_name, evaluator, *args):
    """
    Calls the provided callable, which evaluates to True or False, with the provided arguments.
    If the evaluation results in True add a new feature to the features dictionary with the provided feature name.

    :param feature_dict: the target feature dictionary where the feature should be added
    :param feature_name: the feature name to be used
    :param evaluator: any callable that evaluates to True or False
    :param args: arguments needed for the callable
    :type feature_dict: nalaf.structures.data.FeatureDictionary
    :type feature_name: str
    """
    if evaluator(*args):
        feature_dict[feature_name] = True
