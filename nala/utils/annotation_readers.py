import abc
import glob
import json
import csv
import os
from itertools import chain
from functools import reduce
from operator import lt, gt

from nala.structures.data import Entity, Relation
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

    def __init__(self, directory, read_just_mutations=True, delete_incomplete_docs=True, is_predicted=False, read_relations=False):
        self.directory = directory
        """the directory containing *.ann.json files"""
        self.read_just_mutations = read_just_mutations
        """whether to read in only mutation entities"""
        self.delete_incomplete_docs = delete_incomplete_docs
        """whether to delete documents from the dataset that are not marked as 'anncomplete'"""
        self.is_predicted = is_predicted
        """whether the annotation is predicted or real, which determines where it will be saved"""
        self.read_relations = read_relations
        """whether relations should be read as well"""

    def annotate(self, dataset):
        """
        :type dataset: nala.structures.data.Dataset
        """
        for filename in glob.glob(str(self.directory + "/*.ann.json")):
            with open(filename, 'r', encoding="utf-8") as file:
                try:
                    basename = os.path.basename(filename)
                    if '-' in basename:
                        doc_id = filename.split('-')[-1].replace('.ann.json', '')
                    else:
                        doc_id = basename.replace('.ann.json', '')

                    ann_json = json.load(file)
                    document = dataset.documents[doc_id]
                    if ann_json['anncomplete']:
                        for entity in ann_json['entities']:
                            # Removed 'entity['classId'] == MUT_CLASS_ID', so there is no restriction on entities
                            # if read_just_mutations is False
                            if not self.read_just_mutations:
                                # if it is predicted put it in predicted_annotations else in annotations
                                if self.is_predicted:
                                    ann = Entity(entity['classId'], entity['offsets'][0]['start'],
                                                     entity['offsets'][0]['text'], entity['confidence']['prob'])
                                    document.parts[entity['part']].predicted_annotations.append(ann)
                                else:
                                    ann = Entity(entity['classId'], entity['offsets'][0]['start'],
                                                     entity['offsets'][0]['text'])
                                    document.parts[entity['part']].annotations.append(ann)
                        if self.read_relations:
                            for relation in ann_json['relations']:
                                part = document.parts[relation['entities'][0].split('|')[0]]
                                e1_start = int(relation['entities'][0].split('|')[1].split(',')[0])
                                e2_start = int(relation['entities'][1].split('|')[1].split(',')[0])
                                e1_end = int(relation['entities'][0].split('|')[1].split(',')[1])
                                e2_end = int(relation['entities'][1].split('|')[1].split(',')[1])
                                e1_text = part.text[e1_start:e1_end]
                                e2_text = part.text[e2_start:e2_end]
                                part.relations.append(Relation(e1_start, e2_start, e1_text, e2_text, relation['classId']))
                    elif self.delete_incomplete_docs:
                        del dataset.documents[doc_id]
                    elif self.is_predicted:  # anncomplete=False, delete_incomplete_docs=False, is_predicted=True
                        for entity in ann_json['entities']:
                            if not self.read_just_mutations or entity['classId'] == MUT_CLASS_ID:
                                ann = Entity(entity['classId'], entity['offsets'][0]['start'],
                                                 entity['offsets'][0]['text'], entity['confidence']['prob'])
                                document.parts[entity['part']].predicted_annotations.append(ann)
                except KeyError:
                    # TODO to be removed when external tagtog part_id is fixed, see issue #113
                    pass
        # some docs may not even have a corresponding .ann.json file
        # delete those as well
        if self.delete_incomplete_docs:
            dataset.documents = {doc_id: doc for doc_id, doc in dataset.documents.items()
                                 if sum(len(part.annotations) for part in doc.parts.values()) > 0}


class AnnJsonMergerAnnotationReader(AnnotationReader):
    """
    Merges annotations from several annotators.

    The scheme for merging is:
    1. Start with the annotation from the first annotator
    2. Merge the annotations of the next one, one by one

    There are several available strategies for merging:
    1. Union or intersection
    2. Shortest entity, longest entity or priority
    """
    def __init__(self, directory, strategy='union', entity_strategy='shortest', read_just_mutations=True):
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
        if self.entity_strategy == 'shortest':
            self.operator = lt
        elif self.entity_strategy == 'longest':
            self.operator = gt
        elif self.entity_strategy != 'priority':
            raise ValueError('entity_strategy must be "shortest" or "longest" or "priority"')
        """
        the merging strategy for entities when they are overlapping, can be:
        * longest: takes the longest overlapping entity
        * shortest: takes the shortest overlapping entity
        * priority: take the entity from the annotator with higher priority in the following order:
            (abojchevski, Ectelion, sanjeevkrn, Shpendi, jmcejuela, cuhlig, ANKIT)
        """
        self.priority = ['Ectelion', 'abojchevski', 'sanjeevkrn', 'Shpendi']
        self.read_just_mutations = read_just_mutations
        """whether to read in only mutation entities"""

    def annotate(self, dataset):
        """
        :type dataset: nala.structures.data.Dataset
        """
        if self.entity_strategy == 'priority':
            annotators = self.priority
        else:
            annotators = os.listdir(self.directory)
        self.__merge(dataset, annotators)

    def __append_union(self, merged, entities_x, entities_y):
        # if the strategy is union
        # append the ones that are not overlapping with the already merged ones
        if self.strategy == 'union':
            existing = [Entity(entity['classId'], entity['offsets'][0]['start'], entity['offsets'][0]['text'])
                        for entity in merged]
            for entity in chain(entities_x, entities_y):
                ann = Entity(entity['classId'], entity['offsets'][0]['start'], entity['offsets'][0]['text'])
                if ann not in existing:
                    merged.append(entity)

    def __merge_priority(self, entities_x, entities_y):
        merged = []
        merged_indices_x = []
        merged_indices_y = []

        for index_x, entity_x in enumerate(entities_x):
            for index_y, entity_y in enumerate(entities_y):
                if entity_x['part'] == entity_y['part']:
                    ann_x = Entity(entity_x['classId'], entity_x['offsets'][0]['start'], entity_x['offsets'][0]['text'])
                    ann_y = Entity(entity_y['classId'], entity_y['offsets'][0]['start'], entity_y['offsets'][0]['text'])

                    # if they are the same or overlap
                    # use the first once since that one has higher priority
                    if ann_x == ann_y:
                        if index_x not in merged_indices_x and index_y not in merged_indices_y:
                            merged_indices_x.append(index_x)
                            merged_indices_y.append(index_y)
                            merged.append(entity_x)

        self.__append_union(merged, entities_x, entities_y)
        return merged

    def __merge_pair(self, entities_x, entities_y):
        merged = []
        merged_indices_x = {}
        merged_indices_y = {}

        for index_x, entity_x in enumerate(entities_x):
            for index_y, entity_y in enumerate(entities_y):
                # if they have the same part_id
                if entity_x['part'] == entity_y['part']:
                    ann_x = Entity(entity_x['classId'], entity_x['offsets'][0]['start'], entity_x['offsets'][0]['text'])
                    ann_y = Entity(entity_y['classId'], entity_y['offsets'][0]['start'], entity_y['offsets'][0]['text'])

                    # if they are the same or overlap
                    if ann_x == ann_y:
                        # if neither of them haven't been matched before
                        if index_x not in merged_indices_x and index_y not in merged_indices_y:
                            if self.operator(len(ann_x.text), len(ann_y.text)):
                                merged.append(entity_x)
                                merged_indices_x[index_x] = len(merged), ann_x
                                merged_indices_y[index_y] = len(merged), ann_x
                            else:
                                merged.append(entity_y)
                                merged_indices_x[index_x] = len(merged), ann_y
                                merged_indices_y[index_y] = len(merged), ann_y
                        # if we already matched them before
                        else:
                            # try to see if we have a more suitable match now
                            if index_x in merged_indices_x:
                                index, ann_existing = merged_indices_x[index_x]
                            else:
                                index, ann_existing = merged_indices_y[index_y]
                            if self.operator(len(ann_x.text), len(ann_y.text)):
                                ann_new, entity_new = ann_x, entity_x
                            else:
                                ann_new, entity_new = ann_y, entity_y
                            if self.operator(len(ann_new.text), len(ann_existing.text)):
                                merged[index-1] = entity_new

        self.__append_union(merged, entities_x, entities_y)
        return merged

    def __merge(self, dataset, annotators):
        for doc_id, doc in dataset.documents.items():
            annotator_entities = {}
            # find the annotations that are marked complete by any annotator
            for annotator in annotators:
                # either once or zero times
                for filename in glob.glob(os.path.join(os.path.join(self.directory, annotator), '*{}*.ann.json'.format(doc_id))):
                    with open(filename, 'r', encoding='utf-8') as file:
                        ann_json = json.load(file)
                        if ann_json['anncomplete']:
                            annotator_entities[annotator] = ann_json['entities']

            # if there is at least once set of annotations
            if len(annotator_entities) > 0:
                Entity.equality_operator = 'exact_or_overlapping'
                if self.entity_strategy == 'priority':
                    merged = reduce(self.__merge_priority, [annotator_entities[x] for x in self.priority
                                                            if x in annotator_entities])
                else:
                    merged = reduce(self.__merge_pair, annotator_entities.values())

                for entity in merged:
                    try:
                        part = doc.parts[entity['part']]
                    except KeyError:
                        # TODO: Remove once the tagtog bug is fixed
                        break
                    if not self.read_just_mutations or entity['classId'] == MUT_CLASS_ID:
                        part.annotations.append(
                            Entity(entity['classId'], entity['offsets'][0]['start'], entity['offsets'][0]['text']))
            # delete documents with no annotations
            else:
                del dataset.documents[doc_id]


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
                            ann = Entity(MUT_CLASS_ID, start, row[2])
                            document.parts['abstract'].annotations.append(ann)
                        elif entity_type == 'Gene':
                            ann = Entity('e_1', start, row[2])
                            document.parts['abstract'].annotations.append(ann)
