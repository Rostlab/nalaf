from bs4 import BeautifulSoup
from structures.data import Dataset, Document, Part
import os
import re


class HTMLReader():
    def __init__(self, directory):
        self.directory = directory

    def read(self):
        dataset = Dataset()
        for filename in os.listdir(self.directory):
            with open(os.path.join(self.directory, filename), 'rb') as file:
                soup = BeautifulSoup(file)
                document = Document()

                for part in soup.find_all(id=re.compile('^s')):
                    document.parts[part['id']] = Part(str(part.string))

                dataset.documents[filename.replace('.plain.html', '')] = document
        return dataset