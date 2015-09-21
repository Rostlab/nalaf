import abc


class Tagger():
    """
    Abstract class for external tagger, like tmVar.
    """
    @abc.abstractmethod
    def generate(self, dataset):
        """
        Generates annotations from an external method, like tmVar or SETH.
        :type nala.structures.Dataset:
        :return: new dataset with annotations
        """
        # get annotated documents from somewhere
        return dataset


class TmVarTagger(Tagger):
    """
    TmVar tagger using the RESTApi from "http://www.ncbi.nlm.nih.gov/CBBresearch/Lu/Demo/tmTools/".
    """
    def generate(self, dataset):
        """
        :param dataset: TODO
        :return:
        """
        # todo docset
        # generate pubtator object using PubtatorWriter
        _tmp_pubtator_send = "temp_pubtator_file.txt"

        # submit to tmtools

        # receive new pubtator object

        # parse to dataset object using TmVarReader