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


class NLTKSplitter(Splitter):
    """
    Simple implementation using the function sent_tokenize
    provided by NLTK.

    Implements the abstract class FeatureGenerator.
    """

    def split(self, dataset):
        """
        :type dataset: nalaf.structures.data.Dataset
        """
        for part in dataset.parts():
            part.sentences = sent_tokenize(part.text)
