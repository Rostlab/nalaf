from bs4 import BeautifulSoup
from nala.structures.data import Dataset, Document, Part
import re
import glob


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
