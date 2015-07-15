from bs4 import BeautifulSoup
from nala.structures.data import Dataset, Document, Part, Annotation
import re
import glob
import csv
import os


class HTMLReader:
    def __init__(self, directory):
        self.directory = directory
        """the directory containing the .html files"""

    def read(self):
        """
        read each html file in the directory, parse it and create and instance of Document
        form a dataset consisting of every document parsed and return it

        :returns structures.data.Dataset
        """
        dataset = Dataset()
        filelist = glob.glob(str(self.directory + "/*.html"))
        for filename in filelist:
            with open(filename, 'rb') as file:
                soup = BeautifulSoup(file)
                document = Document()

                for part in soup.find_all(id=re.compile('^s')):
                    document.parts[part['id']] = Part(str(part.string))

                dataset.documents[filename.split('-')[-1].replace('.plain.html', '')] = document
        return dataset


class SETHReader:
    """
    Reader for the SETH-corpus (http://rockt.github.io/SETH/)
    Format: PMID\tabstract (tab separated PMID and abstract)
    """

    def __init__(self, corpus_file):
        self.corpus_file = corpus_file
        """the directory containing the .txt files"""

    def read(self):
        """
        read each .txt file in the directory, parse it and create and instance of Document
        form a dataset consisting of every document parsed and return it

        :returns structures.data.Dataset
        """
        dataset = Dataset()
        with open(self.corpus_file, encoding='utf-8') as file:
            reader = csv.reader(file, delimiter='\t')
            for row in reader:
                document = Document()
                document.parts['abstract'] = Part(row[1])
                dataset.documents[row[0]] = document

        return dataset

class MutationFinderReader:
    """
    Reader for the MutationFinder-corpus (http://mutationfinder.sourceforge.net/)
    Format: PMID\tabstract (tab separated PMID and abstract) in 5 files
    """

    def __init__(self, corpus_folder):
        self.corpus_folder = corpus_folder
        """the directory containing the .txt files"""

    def read(self):
        """
        read each .txt file in the directory, parse it and create and instance of Document
        form a dataset consisting of every document parsed and return it

        :returns structures.data.Dataset
        """
        dataset = Dataset()
        with open(self.corpus_folder, encoding='utf-8') as file:
            reader = csv.reader(file, delimiter='\t')
            for row in reader:
                document = Document()
                document.parts['abstract'] = Part(row[1])
                dataset.documents[row[0]] = document

        return dataset


class VerspoorReader:
    """
    Reader for the Verspoor-corpus (http://www.opennicta.com.au/home/health/variome)
    Format: PMCID-serial-section-paragraph.txt: contains the text from a paragraph of the paper
    """

    def __init__(self, directory):
        self.directory = directory
        """the directory containing the .html files"""

    def read(self):
        """
        read each html file in the directory, parse it and create and instance of Document
        form a dataset consisting of every document parsed and return it

        :returns structures.data.Dataset
        """
        dataset = Dataset()
        file_list = glob.glob(str(self.directory + "/*.txt"))
        for file_path in file_list:
            file_name = os.path.basename(file_path)

            pmid, serial, *_, paragraph, = file_name.replace('.txt', '').split('-')
            # print(pmid, serial, paragraph)

            # for abstract stats generation
            if serial == '01':
                serial = 'abstract'

            with open(file_path, encoding='utf-8') as file:
                text = file.read()
            text = text.replace('** IGNORE LINE **', '')

            if pmid in dataset:
                dataset.documents[pmid].parts[serial + paragraph] = Part(text)
            else:
                document = Document()
                document.parts[serial + paragraph] = Part(text)
                dataset.documents[pmid] = document

        return dataset


class TmVarReader:
    """
    Reader for the tmVar-corpus (http://www.ncbi.nlm.nih.gov/CBBresearch/Lu/Demo/tmTools/#tmVar)

    Note:
        This reader is a bit of an exception as it not only reads the articles,
        but the annotations as well in a single pass. This is due to how the tmVar corpus
        is distributed.


    Format:
        [pmid]|t|[title]
        [pmid]|a|[abstract]
        [pmid]\t[start]\t[end]\t[text]\t[type]\t[normalized]
        [pmid]\t[start]\t[end]\t[text]\t[type]\t[normalized]
        ...

        pmid|t|title
        ...
    """

    def __init__(self, corpus_file):
        self.corpus_file = corpus_file
        """the directory containing the .html files"""

    def read(self):
        dataset = Dataset()
        with open(self.corpus_file, encoding='utf-8') as file:

            for line in file:
                title = line.split('|t|')[-1]
                pmid, abstract = next(file).split('|a|')

                document = Document()
                document.parts['abstract'] = Part(title + abstract)

                line = next(file)
                while line != '\n':
                    _, start, end, text, *_ = line.split('\t')
                    document.parts['abstract'].annotations.append(Annotation('e_2', int(start), text))
                    line = next(file)

                dataset.documents[pmid] = document

        return dataset
