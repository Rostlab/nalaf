import abc
from nltk.tokenize import sent_tokenize


class Splitter():
    @abc.abstractmethod
    def split(self, dataset):
        return


class NTLKSplitter(Splitter):
    def split(self, dataset):
        for document in dataset:
            for part in document:
                part.sentences = sent_tokenize(part.text)