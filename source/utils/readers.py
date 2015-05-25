from bs4 import BeautifulSoup
from structures.data import Dataset, Document, Part
import os
import re


class HTMLReader():
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
        for filename in os.listdir(self.directory):
            with open(os.path.join(self.directory, filename), 'rb') as file:
                soup = BeautifulSoup(file)
                document = Document()

                for part in soup.find_all(id=re.compile('^s')):
                    document.parts[part['id']] = Part(str(part.string))

                dataset.documents[filename.replace('.plain.html', '')] = document
        return dataset