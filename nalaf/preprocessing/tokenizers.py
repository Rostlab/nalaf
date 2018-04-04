import abc
from nalaf.structures.data import Token
import re
from nltk.tokenize import word_tokenize


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
    def tokenize_string(self, string):
        """
        :return string iterable
        """
        return

    @abc.abstractmethod
    def tokenize(self, dataset):
        """
        :type dataset: nalaf.structures.data.Dataset
        """
        return


class GenericTokenizer(Tokenizer):

    def __init__(self, string_splitter_fun):
        self.string_splitter_fun = string_splitter_fun
        "A function that takes a string as input and returns a list/iterator of tokenized string items"


    def tokenize_string(self, string):
        return self.string_splitter_fun(string)


    def tokenize(self, dataset):
        """
        :type dataset: nalaf.structures.data.Dataset
        """
        for part in dataset.parts():
            so_far = 0
            part.sentences = []
            for index, sentence_ in enumerate(part.sentences_):
                part.sentences.append([])

                for token_word in self.tokenize_string(sentence_):
                    token_start = part.text.find(token_word, so_far)
                    so_far = token_start + len(token_word)
                    part.sentences[index].append(Token(token_word, token_start))




NLTK_TOKENIZER = GenericTokenizer(word_tokenize)


class TmVarTokenizer(Tokenizer):
    """
    Implementation of the TmVar tokenizer as descriped in their paper. Code ported from perl.
    Requires
    """

    def tokenize_string(self, string):
        sentence = string
        sentence = re.sub('([0-9])([A-Za-z])', r'\1 \2', sentence)
        # removing for now split from Capital to Lower
        # no noticeable difference in performance but we keep whole words
        # sentence = re.sub('([A-Z])([a-z])', r'\1 \2', sentence)
        sentence = re.sub('([a-z])([A-Z])', r'\1 \2', sentence)
        sentence = re.sub('([A-Za-z])([0-9])', r'\1 \2', sentence)
        sentence = re.sub('([a-z])(fs)', r'\1 \2', sentence)

        # separate non-ascii characters into their own tokens
        sentence = re.sub('([^\x00-\x7F])', r' \1 ', sentence)

        sentence = re.sub('([\W\-_])', r' \1 ', sentence)

        return sentence.split()  # splits by white space


    def tokenize(self, dataset):
        """
        :type dataset: nalaf.structures.data.Dataset
        """
        for part in dataset.parts():
            so_far = 0
            part.sentences = []
            for index, sentence_ in enumerate(part.sentences_):
                part.sentences.append([])

                for token_word in self.tokenize_string(sentence_):
                    token_start = part.text.find(token_word, so_far)
                    so_far = token_start + len(token_word)
                    part.sentences[index].append(Token(token_word, token_start))
