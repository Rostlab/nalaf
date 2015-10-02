import abc
import glob
import json
from nala.structures.data import Annotation
import csv
import os
from nala.utils import MUT_CLASS_ID


class AnnotationReader:
    """
    Abstract class for annotating the dataset.
    Subclasses that inherit this class should:
    * Be named [Name]AnnotationReader
    * Implement the abstract method annotate
    * Append new items to the list field "annotations" of each Part in the dataset
    """

    @abc.abstractmethod
    def annotate(self, dataset):
        """
        :type dataset: nala.structures.data.Dataset
        """
        return


class AnnJsonAnnotationReader(AnnotationReader):
    """
    Reads the annotations from .ann.json format.

    Implements the abstract class Annotator.
    """

    def __init__(self, directory, read_just_mutations=True, delete_incomplete_docs=True):
        self.directory = directory
        """the directory containing *.ann.json files"""
        self.read_just_mutations = read_just_mutations
        """whether to read in only mutation entities"""
        self.delete_incomplete_docs = delete_incomplete_docs
        """whether to delete documents from the dataset that are not marked as 'anncomplete'"""

    def annotate(self, dataset):
        """
        :type dataset: nala.structures.data.Dataset
        """
        from nala.utils import MUT_CLASS_ID
        for filename in glob.glob(str(self.directory + "/*.ann.json")):
            with open(filename, 'r', encoding="utf-8") as file:
                try:
                    basename = os.path.basename(filename)
                    if '-' in basename:
                        doc_id = filename.split('-')[-1].replace('.ann.json', '')
                    else:
                        doc_id = basename.replace('.ann.json', '')

                    ann_json = json.load(file)
                    if ann_json['anncomplete']:
                        document = dataset.documents[doc_id]
                        for entity in ann_json['entities']:
                            if not self.read_just_mutations or entity['classId'] == MUT_CLASS_ID:
                                ann = Annotation(entity['classId'], entity['offsets'][0]['start'], entity['offsets'][0]['text'])
                                document.parts[entity['part']].annotations.append(ann)
                    elif self.delete_incomplete_docs:
                        del dataset.documents[doc_id]
                    else:
                        document = dataset.documents[doc_id]
                        for entity in ann_json['entities']:
                            if not self.read_just_mutations or entity['classId'] == MUT_CLASS_ID:
                                ann = Annotation(entity['classId'], entity['offsets'][0]['start'], entity['offsets'][0]['text'], entity['confidence']['prob'])
                                document.parts[entity['part']].predicted_annotations.append(ann)
                except KeyError:
                    # TODO to be removed when external tagtog part_id is fixed, see issue #113
                    pass


class AnnJsonMerger:
    """
    Merges annotations from several annotators.

    The scheme for merging is:
    1. Start with the annotation from the first annotator
    2. Merge the annotations of the next one, one by one

    There are several available strategies for merging:
    1. Union or intersection
    2. Shortest or longest entity
    """
    def __init__(self, directory, strategy='union', entity_strategy='longest', read_just_mutations=True):
        self.directory = directory
        """
        the directory containing several sub-directories with .ann.json files
        corresponding to the annotations of each annotator
        """
        self.strategy = strategy
        """
        the general merging strategy, can be:
        * intersection: only merge overlapping entities
        * union: merge not existing entities as well
        """
        self.entity_strategy = entity_strategy
        """
        the merging strategy for entities when they are overlapping, can be:
        * longest: takes the longest overlapping entity
        * shortest: takes the shortest overlapping entity
        """
        self.read_just_mutations = read_just_mutations
        """whether to read in only mutation entities"""

    def merge(self, dataset):
        """
        :type dataset: nala.structures.data.Dataset
        """
        annotators = os.listdir(self.directory)
        AnnJsonAnnotationReader(os.path.join(self.directory, annotators[0])).annotate(dataset)

        for annotator in annotators[1:]:
            self.merge_annotations_into_dataset(dataset, os.path.join(self.directory, annotator))

        # remove duplicates that might have been added
        # in the case when one ann spans across several others
        if self.entity_strategy == 'longest':
            Annotation.equality_operator = 'exact'
            for part in dataset.parts():
                to_be_removed = []
                parsed = []
                for index, ann in enumerate(part.annotations):
                    if ann in parsed:
                        to_be_removed.append(index)
                    else:
                        parsed.append(ann)
                part.annotations = [ann for index, ann in enumerate(part.annotations) if index not in to_be_removed]

    def merge_annotations_into_dataset(self, dataset, annotations_directory):
        for doc_id, document in dataset.documents.items():
            # either once or zero times
            for filename in glob.glob(os.path.join(annotations_directory, '*{}*.ann.json'.format(doc_id))):
                with open(filename, 'r', encoding='utf-8') as file:
                    ann_json = json.load(file)
                    if ann_json['anncomplete']:
                        for entity in ann_json['entities']:
                            if not self.read_just_mutations or entity['classId'] == MUT_CLASS_ID:
                                ann = Annotation(entity['classId'], entity['offsets'][0]['start'], entity['offsets'][0]['text'])
                                try:
                                    for index, existing_ann in enumerate(document.parts[entity['part']].annotations):
                                        Annotation.equality_operator = 'overlapping'
                                        if ann == existing_ann:
                                            if self.entity_strategy == 'shortest':
                                                if len(ann.text) < len(existing_ann.text):
                                                    document.parts[entity['part']].annotations[index] = ann
                                            elif self.entity_strategy == 'longest':
                                                if len(ann.text) > len(existing_ann.text):
                                                    document.parts[entity['part']].annotations[index] = ann
                                            else:
                                                raise ValueError('entity_strategy must be "shortest" or "longest"')

                                    # annotations not in original dataset
                                    # include only if the strategy is union
                                    Annotation.equality_operator = 'exact_or_overlapping'
                                    if self.strategy == 'union' and ann not in document.parts[entity['part']].annotations:
                                        document.parts[entity['part']].annotations.append(ann)
                                except KeyError:
                                    pass


class SETHAnnotationReader(AnnotationReader):
    """
    Reads the annotations from the SETH-corpus (http://rockt.github.io/SETH/)
    Format: filename = PMID
    Tab separated:
        annotation_type = T# or R# or E# (entity or relationship or ?)
        mention = space separated:
            entity_type start end
        text

        entity_type = one of the following: 'SNP', 'Gene', 'RS'

    We map:
        SNP to e_2 (mutation entity)
        Gene to e_1 (protein entity)
        RS to e_2 (mutation entity)

    Implements the abstract class Annotator.
    """

    def __init__(self, directory):
        self.directory = directory
        """the directory containing *.ann files"""

    def annotate(self, dataset):
        """
        :type dataset: nala.structures.data.Dataset
        """
        for filename in glob.glob(str(self.directory + "/*.ann")):
            with open(filename, 'r', encoding='utf-8') as file:
                reader = csv.reader(file, delimiter='\t')

                pmid = os.path.basename(filename).replace('.ann', '')
                document = dataset.documents[pmid]

                for row in reader:
                    if row[0].startswith('T'):
                        entity_type, start, end = row[1].split()

                        if entity_type == 'SNP' or entity_type == 'RS':
                            ann = Annotation(MUT_CLASS_ID, start, row[2])
                            document.parts['abstract'].annotations.append(ann)
                        elif entity_type == 'Gene':
                            ann = Annotation('e_1', start, row[2])
                            document.parts['abstract'].annotations.append(ann)
