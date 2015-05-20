from bs4 import BeautifulSoup
from structures.data import Document, Part
import os


class HTMLReader():
    def __init__(self, directory):
        self.directory = directory

    def read(self):
        dataset = []
        for filename in os.listdir(self.directory)[6:7]:
            with open(os.path.join(self.directory, filename), 'rb') as file:
                soup = BeautifulSoup(file)
                document = Document(filename)
                document.parts = [Part(str(part.string)) for part in soup.find_all('p')]
                dataset.append(document)
        return dataset