import abc
import os
import json
from nala.structures.data import Annotation


class Annotator:
    """
    Abstract class for annotating the dataset.
    Subclasses that inherit this class should:
    * Be named [Name]Annotator
    * Implement the abstract method annotate
    * Append new items to the list field "annotations" of each Part in the dataset
    """
    @abc.abstractmethod
    def annotate(self, dataset):
        """
        :type dataset: structures.data.Dataset
        """
        return


class ReadFromAnnJsonAnnotator(Annotator):
    """
    Reads the annotations from .ann.json format.

    Implements the abstract class Annotator.
    """
    def __init__(self, directory):
        self.directory = directory
        """the directory containing *.ann.json files"""

    def annotate(self, dataset):
        """
        :type dataset: structures.data.Dataset
        """
        for filename in os.listdir(self.directory):
            with open(os.path.join(self.directory, filename)) as file:
                try:
                    document = dataset.documents[filename.replace('.ann.json', '')]
                    ann_json = json.load(file)
                    for entity in ann_json['entities']:
                        ann = Annotation(entity['classId'], entity['offsets'][0]['start'], entity['offsets'][0]['text'])
                        document.parts[entity['part']].annotations.append(ann)
                except KeyError:
                    pass
