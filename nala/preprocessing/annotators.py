import abc
import glob
import os
import json
import requests
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
        for filename in glob.glob(str(self.directory + "/*.ann.json")):
            with open(filename, 'r', encoding="utf-8") as file:
                try:
                    document = dataset.documents[filename.split('-')[-1].replace('.ann.json', '')]
                    ann_json = json.load(file)
                    for entity in ann_json['entities']:
                        ann = Annotation(entity['classId'], entity['offsets'][0]['start'], entity['offsets'][0]['text'])
                        document.parts[entity['part']].annotations.append(ann)
                except KeyError:
                    pass
