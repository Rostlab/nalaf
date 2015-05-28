import abc


class NLDefiner():
    """
    Abstract class for determining whether an annotation in the dataset is a natural language (NL) mention.
    Subclasses that inherit this class should:
    * Be named [Name]NLDefiner
    * Implement the abstract method define
    * Set the value
    """
    @abc.abstractmethod
    def define(self, dataset):
        """
        :type dataset: structures.data.Dataset
        """
        return


class TestNLDefiner(NLDefiner):
    """
    TODO: Explain the general thing here


    Requires the list field "annotations" to be previously set.
    Implements the abstract class NLDefiner.
    """
    def __init__(self, min_spaces=3, max_length=22):
        self.min_spaces = min_spaces
        self.max_length = max_length

    def define(self, dataset):
        for ann in dataset.annotations():
            if ann.class_id == 'e_2' \
                    and len(ann.text) < self.max_length \
                    and len(ann.text.split()) >= self.min_spaces:
                ann.is_nl = True


class InclusiveNLDefiner(NLDefiner):

    def __init__(self, min_length=18):
        self.min_spaces = 3
        self.min_length = min_length

    def define(self, dataset):
        for ann in dataset.annotations():
            if ann.class_id == 'e_2' \
                    and len(ann.text) >= self.min_length \
                    and len(ann.text.split()) >= self.min_spaces:
                ann.is_nl = True
