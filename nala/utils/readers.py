from bs4 import BeautifulSoup
from nala.structures.data import Dataset, Document, Part
import re
import glob
import csv


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
