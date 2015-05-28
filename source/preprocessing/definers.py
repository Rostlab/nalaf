import abc
import json
import re

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


class InclusiveNLDefiner(NLDefiner):

    def __init__(self, min_length=18):
        self.min_spaces = 3
        self.min_length = min_length

    def define(self, dataset):
        for ann in dataset.annotations():
            if ann.class_id == 'e_2' \
                    and len(ann.text) >= self.min_length \
                    and len(ann.text.split(" ")) > self.min_spaces:
                ann.is_nl = True


class ExclusiveNLDefiner(NLDefiner):

    """docstring for ExclusiveNLDefiner"""

    def __init__(self):
        self.max_spaces = 2
        self.conventions_file = 'regex_st.json'

        # read in file regex_st.json into conventions array
        with open(self.conventions_file, 'r') as f:
            self.conventions = json.loads(f.read())

    def define(self, dataset):
        for ann in dataset.annotations():
            if ann.class_id == 'e_2' \
                    and len(ann.text.split(" ")) <= self.max_spaces:
                for conv_re in conventions:
                    if re.search(conv_re):
                        # TODO continue here (1)
                        pass
                ann.is_nl = True

#
#
# END OF FILE
