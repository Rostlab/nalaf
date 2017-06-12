import abc
from nltk.tokenize import sent_tokenize


class Splitter:
    """
    Abstract class for splitting the raw text (or tokens if Tokenizer was called first)
    into sentences for each document in the dataset.
    Subclasses that inherit this class should:
    * Be named [Name]Splitter
    * Implement the abstract method split
    * Append new items to the list field "sentences" of each Part in the dataset
    """

    @abc.abstractmethod
    def split(self, dataset):
        """
        :type dataset: nalaf.structures.data.Dataset
        """
        return


class GenericSplitter(Splitter):

    def __init__(self, string_splitter_fun):
        self.string_splitter_fun = string_splitter_fun
        "A function that takes a string as input and returns a list/iterator of splitted string items"


    def split(self, dataset):
        """
        :type dataset: nalaf.structures.data.Dataset
        """
        for part in dataset.parts():
            part.sentences_ = list(self.string_splitter_fun(part.text))


NLTK_SPLITTER = GenericSplitter(sent_tokenize)
"""
Simple implementation using the function NLTK::sent_tokenize.
"""


class NLTKSplitter(Splitter):
    import warnings
    warnings.warn('Use `NLTK_SPLITTER` instead', DeprecationWarning)

    def split(self, dataset):
        NLTK_SPLITTER.split(dataset)
