import abc
from nltk.tokenize import word_tokenize
from structures.data import Token


class Tokenizer():
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
        :type dataset: structures.data.Dataset
        """
        return


class NLTKTokenizer(Tokenizer):
    def tokenize(self, dataset):
        """
        :type dataset: structures.data.Dataset
        """
        for part in dataset.parts():
                part.sentences = [[Token(word) for word in word_tokenize(sentence)] for sentence in part.sentences]