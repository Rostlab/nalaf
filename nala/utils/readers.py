from bs4 import BeautifulSoup
from nala.structures.data import Dataset, Document, Part
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


class VerspoorReader:
    """
    Reader for the Verspoor-corpus (http://www.opennicta.com.au/home/health/variome)
    Format: PMCID-serial-section-paragraph.txt: contain the text from a paragraph of the paper
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

            with open(file_path) as file:
                text = file.read()
            text = text.replace('** IGNORE LINE **', '')

            if pmid in dataset:
                dataset.documents[pmid].parts[serial + paragraph] = Part(text)
            else:
                document = Document()
                document.parts[serial + paragraph] = Part(text)
                dataset.documents[pmid] = document

        return dataset
