import abc
import os
import json
from structures.data import Annotation

class Annotator():
    @abc.abstractmethod
    def annotate(self, dataset):
        return


class ReadFromAnnJsonAnnotator(Annotator):
    def __init__(self, directory):
        self.directory = directory

    def annotate(self, dataset):
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
