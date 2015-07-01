import abc
import glob
import json
from nala.structures.data import Annotation
import csv
import os


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
        :type dataset: structures.data.Dataset
        """
        return


class AnnJsonAnnotationReader(AnnotationReader):
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
        RS is skipped

    Implements the abstract class Annotator.
    """

    def __init__(self, directory):
        self.directory = directory
        """the directory containing *.ann files"""

    def annotate(self, dataset):
        """
        :type dataset: structures.data.Dataset
        """
        for filename in glob.glob(str(self.directory + "/*.ann")):
            with open(filename, 'r', encoding='utf-8') as file:
                reader = csv.reader(file, delimiter='\t')

                pmid = os.path.basename(filename).replace('.ann', '')
                document = dataset.documents[pmid]

                for row in reader:
                    if row[0].startswith('T'):
                        entity_type, start, end = row[1].split()

                        if entity_type == 'SNP':
                            ann = Annotation('e_2', start, row[2])
                            document.parts['abstract'].annotations.append(ann)
                        elif entity_type == 'Gene':
                            ann = Annotation('e_1', start, row[2])
                            document.parts['abstract'].annotations.append(ann)


class VerspoorAnnotationReader(AnnotationReader):
    """
    Reader for the Verspoor-corpus (http://www.opennicta.com.au/home/health/variome)

    Format: PMCID-serial-section-paragraph.ann: contains the standoff annotation of the paragraph
    Tab separated:
        annotation_type = T# or R# (entity or relationship)
        mention = space separated:
            entity_type start end
        text

        entity_type = one of the following: 'gene', 'Disorder', 'Concepts_Ideas', 'Physiology', 'Phenomena',
        'mutation', 'disease', 'age', 'size', 'ethnicity', 'cohort-patient', 'gender', 'body-part'

    We map:
        mutation to e_2 (mutation entity)
        gene to e_1 (protein entity)
        the others are skipped

    Implements the abstract class Annotator.
    """

    def __init__(self, directory):
        self.directory = directory
        """the directory containing *.ann.json files"""

    def annotate(self, dataset):
        """
        :type dataset: structures.data.Dataset
        """
        file_list = glob.glob(str(self.directory + "/*.ann"))
        for file_path in file_list:
            file_name = os.path.basename(file_path)

            pmid, serial, *_, paragraph, = file_name.replace('.ann', '').split('-')

            with open(file_path, 'r', encoding='utf-8') as file:
                reader = csv.reader(file, delimiter='\t')

                for row in reader:
                    if row[0].startswith('T'):
                        entity_type, start, end = row[1].split()

                        if entity_type == 'mutation':
                            ann = Annotation('e_2', start, row[2])
                            dataset.documents[pmid].parts[serial + paragraph].annotations.append(ann)
                        elif entity_type == 'gene':
                            ann = Annotation('e_1', start, row[2])
                            dataset.documents[pmid].parts[serial + paragraph].annotations.append(ann)
