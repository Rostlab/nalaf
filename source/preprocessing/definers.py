import abc
import csv
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


class TmVarRegexNLDefiner(NLDefiner):
    """
    Definier based on tmVar regexes
    #TODO: Document it better
    """
    def define(self, dataset):
        with open('RegEx.NL') as file:
            regexps = list(csv.reader(file, delimiter='\t'))

        compiled_regexps = []
        for regex in regexps:
            if regex[0].startswith('(?-xism:'):
                try:
                    compiled_regexps.append(re.compile(regex[0].replace('(?-xism:', ''),
                                                       re.VERBOSE | re.IGNORECASE | re.DOTALL | re.MULTILINE))
                except:
                    pass
            else:
                compiled_regexps.append(re.compile(regex[0]))

        for ann in dataset.annotations():
            if ann.class_id == 'e_2':
                matches = [regex.match(ann.text) for regex in compiled_regexps]
                if not any(matches):
                    ann.is_nl = True