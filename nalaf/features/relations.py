import abc
from nalaf.features import FeatureGenerator
from nltk.stem import PorterStemmer
import re


class EdgeFeatureGenerator(FeatureGenerator):
    """
    Abstract class for generating features for each edge in the dataset.
    Subclasses that inherit this class should:
    * Be named [Name]FeatureGenerator
    * Implement the abstract method generate
    * Append new items to the dictionary field "features" of each Edge in the dataset
    """

    @abc.abstractmethod
    def generate(self, dataset, feature_set, is_training_mode):
        """
        :type dataset: nalaf.structures.data.Dataset
        """
        pass


    def add_to_feature_set(self, feature_set, is_training_mode, edge, feature_name, value=1):
        """
        Return True if feature was added to feature_set. False, otherwise

        If the feature_name is None, the feature is not added in anycase. See: self.mk_feature_name
        """
        if feature_name is None:
            return False

        else:
            if not re.search('\[-?[0-9]+\]$', feature_name):
                feature_name = feature_name + "_[0]"  # See logic of definition in: FeatureDictionary

            if is_training_mode:
                if feature_name not in feature_set.keys():
                    feature_set[feature_name] = len(feature_set.keys()) + 1
                edge.features[feature_set[feature_name]] = value
                return True
            else:
                if feature_name in feature_set.keys():
                    edge.features[feature_set[feature_name]] = value
                    return True
            else:
                return False


    def mk_feature_name(self, prefix, *args):
        if prefix is None:
            return None
        else:
            l = [str(x) for x in ([prefix] + list(args))]
            return "_".join(l)


# -----------------------------------------------------------------------------

class NamedEntityCountFeatureGenerator(EdgeFeatureGenerator):
    """
    Generates Named Entity Count for each sentence that contains an edge
    """

    def __init__(self, entity_type, prefix1):
        self.entity_type = entity_type
        """type of entity"""
        self.prefix1 = prefix1


    def generate(self, dataset, feature_set, is_training_mode):
        for edge in dataset.edges():
            entities = edge.part.get_entities_in_sentence(edge.sentence_id, self.entity_type)
            num_entities = len(entities)
            feature_name = self.mk_feature_name(self.prefix1, self.entity_type, 'count', num_entities)
            self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name, value=1)


# TODO rest is potential material to delete, almost exact copy in relna's sentence.py

class BagOfWordsFeatureGenerator(FeatureGenerator):
    """
    Generates Bag of Words representation for each sentence that contains an edge

    :type feature_set: nalaf.structures.data.FeatureDictionary
    :type training_mode: bool
    """
    def __init__(self, feature_set, training_mode=True):
        self.feature_set = feature_set
        """the feature set for the dataset"""
        self.training_mode = training_mode
        """whether the mode is training or testing"""

    def generate(self, dataset):
        for edge in dataset.edges():
            sentence = edge.part.sentences[edge.sentence_id]
            if self.training_mode:
                for token in sentence:
                    feature_name = 'bow_' + token.word + '[0]'
                    if feature_name not in self.feature_set:
                        self.feature_set[feature_name] = len(self.feature_set.keys())+1
                    edge.features[self.feature_set[feature_name]] = 1
            else:
                for token in sentence:
                    feature_name = 'bow_' + token.word + '[0]'
                    if feature_name in self.feature_set.keys():
                        edge.features[self.feature_set[feature_name]] = 1

class StemmedBagOfWordsFeatureGenerator(FeatureGenerator):
    """
    Generates stemmed Bag of Words representation for each sentence that contains
    an edge, using the function given in the argument.

    By default it uses Porter stemmer

    :type feature_set: nalaf.structures.data.FeatureDictionary
    :type training_mode: bool
    """

    def __init__(self, feature_set, training_mode=True):
        self.feature_set = feature_set
        """the feature set for the dataset"""
        self.training_mode = training_mode
        """whether the mode is training or testing"""
        self.stemmer = PorterStemmer()

    def generate(self, dataset):
        for edge in dataset.edges():
            sentence = edge.part.sentences[edge.sentence_id]
            if self.training_mode:
                for token in sentence:
                    feature_name = 'bow_stem_' + self.stemmer.stem(token.word) + '[0]'
                    if feature_name not in self.feature_set:
                        self.feature_set[feature_name] = len(self.feature_set.keys())+1
                    edge.features[self.feature_set[feature_name]] = 1
            else:
                for token in sentence:
                    feature_name = 'bow_stem_' + self.stemmer.stem(token.word) + '[0]'
                    if feature_name in self.feature_set.keys():
                        edge.features[self.feature_set[feature_name]] = 1

class OrderOfEntitiesFeatureGenerator(FeatureGenerator):
    """
    Generates the order in which the entities are present in the sentence.
    That is, whether it is '...entity1...entity2...' or '...entity2...entity1...'

    Value of 1 means that the order is '...entity1...entity2...'
    Value of 0 means that the order is '...entity2...entity1...'

    :type feature_set: nalaf.structures.data.FeatureDictionary
    :type training_mode: bool
    """
    def __init__(self, feature_set, training_mode=True):
        self.feature_set = feature_set
        """the feature set for the dataset"""
        self.training_mode = training_mode
        """whether the mode is training or testing"""

    def generate(self, dataset):
        for edge in dataset.edges():
            feature_name = 'order_entities_[0]'
            if self.training_mode:
                if feature_name not in self.feature_set.keys():
                    self.feature_set[feature_name] = len(self.feature_set.keys())+1
                if edge.entity1.offset < edge.entity2.offset:
                    edge.features[self.feature_set[feature_name]] = 1
            else:
                if feature_name in self.feature_set.keys():
                    if edge.entity1.offset < edge.entity2.offset:
                        edge.features[self.feature_set[feature_name]] = 1


class CapitalizedTokenFeatureGenerator(FeatureGenerator):
    """
    Checks if the entity in the edge is capitalized or not.

    Value of 1 means that the entity is capitalized
    Value of 0 means that the entity is not capitalized

    :type feature_set: nalaf.structures.data.FeatureDictionary
    :type training_mode: bool
    """
    def __init__(self, feature_set, training_mode=True):
        self.feature_set = feature_set
        """the feature set for the dataset"""
        self.training_mode = training_mode
        """whether the mode is training or testing"""

    def generate(self, dataset):
        feature_name_1 = 'entity_1_capitalized_[0]'
        feature_name_2 = 'entity_2_capitalized_[0]'
        if self.training_mode:
            if feature_name_1 not in self.feature_set.keys():
                self.feature_set[feature_name_1] = len(self.feature_set.keys()) + 1
            if feature_name_2 not in self.feature_set.keys():
                self.feature_set[feature_name_2] = len(self.feature_set.keys()) + 1
            for edge in dataset.edges():
                if edge.entity1.text.isupper():
                    edge.features[self.feature_set[feature_name_1]] = 1
                if edge.entity2.text.isupper():
                    edge.features[self.feature_set[feature_name_2]] = 1
        else:
            for edge in dataset.edges():
                if feature_name_1 in self.feature_set.keys():
                    if edge.entity1.text.isupper():
                        edge.features[self.feature_set[feature_name_1]] = 1
                if feature_name_2 in self.feature_set.keys():
                    if edge.entity2.text.isupper():
                        edge.features[self.feature_set[feature_name_2]] = 1


class WordFilterFeatureGenerator(FeatureGenerator):
    """
    Checks if the sentence containing an edge contains any of the words
    given in the list.

    Value of 1 means that the sentence contains that word
    Value of 0 means that the sentence does not contain the word

    :type feature_set: nalaf.structures.data.FeatureDictionary
    :type words: list[str]
    :type stem: bool
    :type training_mode: bool
    """
    def __init__(self, feature_set, words, stem=True, training_mode=True):
        self.feature_set = feature_set
        """the feature set for the dataset"""
        self.words = words
        """a list of words to check for their presence in the sentence"""
        self.stem = stem
        """whether the words in the sentence and the list should be stemmed"""
        self.training_mode = True
        """whether the mode is training or testing"""
        self.stemmer = PorterStemmer()

    def generate(self, dataset):
        if self.stem:
            stemmed_words = [ self.stemmer.stem(word) for word in self.words ]
            if self.training_mode:
                for edge in dataset.edges():
                    sentence = edge.part.sentences[edge.sentence_id]
                    for token in sentence:
                        if self.stemmer.stem(token.word) in stemmed_words:
                            feature_name = 'word_filter_stem_' + self.stemmer.stem(token.word) + '[0]'
                            if feature_name not in self.feature_set.keys():
                                self.feature_set[feature_name] = len(self.feature_set.keys()) + 1
                            edge.features[self.feature_set[feature_name]] = 1
            else:
                for edge in dataset.edges():
                    sentence = edge.part.sentences[edge.sentence_id]
                    for token in sentence:
                        if self.stemmer.stem(token.word) in stemmed_words:
                            feature_name = 'word_filter_stem_' + self.stemmer.stem(token.word) + '[0]'
                            if feature_name in self.feature_set.keys():
                                edge.features[self.feature_set[feature_name]] = 1
        else:
            if self.training_mode:
                for edge in dataset.edges():
                    sentence = edge.part.sentences[edge.sentence_id]
                    for token in sentence:
                        if token.word in self.words:
                            feature_name = 'word_filter_' + token.word + '[0]'
                            if feature_name not in self.feature_set.keys():
                                self.feature_set[feature_name] = len(self.feature_set.keys()) + 1
                            edge.features[self.feature_set[feature_name]] = 1
            else:
                for edge in dataset.edges():
                    sentence = edge.part.sentences[edge.sentence_id]
                    for token in sentence:
                        if token in self.words:
                            feature_name = 'word_filter_' + token.word + '[0]'
                            if feature_name in self.feature_set.keys():
                                edge.features[self.feature_set[feature_name]] = 1


class NPChunkRootFeatureGenerator(FeatureGenerator):
    """
    Generate Noun Phrase Chunks for each sentence containing an edge and store
    the roots of each noun phrase chunk

    This requires the Python module SpaCy (http://spacy.io/) and NumPy. SpaCy
    can be installed by `pip3 install spacy`. Additionally it also requires
    the package data, which can be downloaded using
    `python3 -m spacy.en.download all`

    :type feature_set: nalaf.structures.data.FeatureDictionary
    :type nlp: spacy.en.English
    :type training_mode: bool
    """
    def __init__(self, feature_set, nlp, training_mode=True):
        self.feature_set = feature_set
        """the feature set for the dataset"""
        self.nlp = nlp
        """an instance of spacy.en.English"""
        self.training_mode = training_mode
        """whether the mode is training or testing"""

    def generate(self, dataset):
        if self.training_mode:
            for edge in dataset.edges():
                sent = edge.part.get_sentence_string_array()[edge.sentence_id]
                sentence = self.nlp(sent, parse=True, tag=True)
                for chunk in sentence.noun_chunks:
                    feature_name = 'np_chunk_root_' + chunk.root.orth_ + '[0]'
                    if feature_name not in self.feature_set.keys():
                        self.feature_set[feature_name] = len(self.feature_set.keys()) + 1
                    edge.features[self.feature_set[feature_name]] = 1
        else:
            for edge in dataset.edges():
                sent = edge.part.get_sentence_string_array()[edge.sentence_id]
                sentence = self.nlp(sent, parse=True, tag=True)
                for chunk in sentence.noun_chunks:
                    feature_name = 'np_chunk_root_' + chunk.root.orth_ + '[0]'
                    if feature_name in self.feature_set.keys():
                        edge.features[self.feature_set[feature_name]] = 1
