import abc
from nltk.tokenize import sent_tokenize


class Splitter():
    @abc.abstractmethod
    def split(self, dataset):
        return


class NTLKSplitter(Splitter):
    def split(self, dataset):
        for part in dataset.parts():
                part.sentences = sent_tokenize(part.text)