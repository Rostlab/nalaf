import abc
from nltk.tokenize import word_tokenize
from nala.structures.data import Token
import re


class Tokenizer:
    """
    Abstract class for splitting the raw text (or sentences if Splitter was called first)
    into tokens for each document in the dataset.
    Subclasses that inherit this class should:
    * Be named [Name]Tokenizer
    * Implement the abstract method tokenize
    * Append new sub-items to each list of the list field "sentences" of each Part in the dataset
    """

    @abc.abstractmethod
    def tokenize(self, dataset):
        """
        :type dataset: nala.structures.data.Dataset
        """
        return


class NLTKTokenizer(Tokenizer):
    def tokenize(self, dataset):
        """
        :type dataset: nala.structures.data.Dataset
        """
        for part in dataset.parts():
            part.sentences = [[Token(word) for word in word_tokenize(sentence)] for sentence in part.sentences]


class TmVarTokenizer(Tokenizer):
    """
    Implementation of the TmVar tokenizer as descriped in their paper. Code ported from perl.
    Requires
    """
    def tokenize(self, dataset):
        """
        :type dataset: nala.structures.data.Dataset
        """
        for part in dataset.parts():
            for index, sentence in enumerate(part.sentences):
                sentence = re.sub('([0-9])([A-Za-z])', r'\1 \2', sentence)
                sentence = re.sub('[^ ]([A-Z])([a-z])', r'\1 \2', sentence)
                sentence = re.sub('([a-z])([A-Z])', r'\1 \2', sentence)
                sentence = re.sub('([A-Za-z])([0-9])', r'\1 \2', sentence)
                sentence = re.sub('([a-z])(fs)', r'\1 \2', sentence)
                sentence = re.sub('([\W\-_])', r' \1 ', sentence)

                part.sentences[index] = [Token(token) for token in sentence.split()]#18
