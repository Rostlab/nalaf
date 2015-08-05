import json
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
            # serial = first section
            # paragrahp = second section
            # divided by newlines = 3rd param which is p## or h##
            # NOTE h## follows with p## so no seperate calculations

            # print(pmid, serial, paragraph)

            with open(file_path, encoding='utf-8') as file:
                text_raw = file.read()

            text = text_raw.replace('** IGNORE LINE **\n', '')
            paragraph_list = text.split('\n\n')

            # inital offset for raw_text
            tot_offset = text_raw.count('** IGNORE LINE **\n') * 18
            offsets = [tot_offset]

            for i, text_part in enumerate(paragraph_list):
                # if text is empty (usually last text due to splitting of "\n\n")
                if text_part != "":
                    pass
                    first_serial = "s" + serial
                    second_serial = "s" + paragraph.replace("p","")

                    # OPTIONAL to activate but annotation reader has to modified as well
                    # header when 10 >= words in text_part
                    # if len(text_part.split(" ")) < 10:
                    #     paragraph_id = "h" + "{:02d}".format(i + 1)
                    # else:
                    paragraph_id = "p" + "{:02d}".format(i + 1)

                    partid = "{0}{1}{2}".format(first_serial, second_serial, paragraph_id)

                    if pmid in dataset:
                        dataset.documents[pmid].parts[partid] = Part(text_part)
                    else:
                        document = Document()
                        document.parts[partid] = Part(text_part)
                        dataset.documents[pmid] = document

                    # add offset for next paragraph
                    tot_offset += len(text_part) + 2
                    offsets.append(tot_offset)

            # to delete last element
            del offsets[-1]
            # print(pmid, serial, paragraph, offsets)

            # annotations
            with open(file_path.replace('.txt', '.ann'), encoding='utf-8') as f:
                reader = csv.reader(f, delimiter='\t')
                for row in reader:
                        if row[0].startswith('T'):
                            entity_type, start, end = row[1].split()
                            # start = int(start)
                            # calc of which part it belongs to
                            id = "None"
                            to_correct_off = 0

                            last_off = 0

                            for i, off in enumerate(offsets):
                                if int(start) < off:
                                    id = "p{:02d}".format(i)
                                    to_correct_off = last_off
                                    break
                                last_off = off

                            # if last element
                            if id == "None":
                                id = "p{:02d}".format(len(offsets))
                                to_correct_off = offsets[-1]

                            properid = "{0}{1}{2}".format(first_serial, second_serial, id)

                            if id == "None":
                                print("None???", pmid, serial, paragraph, start, offsets)

                            if (first_serial + second_serial + id) not in dataset.documents[pmid].parts:
                                print("NoKEY???", dataset.documents[pmid].parts)
                                # print(json.dumps(document, indent=4))

                            # print(document.parts[properid].text[int(start) - to_correct_off:int(start) - to_correct_off + len(row[2])], "==", row[2])


                            calc_ann_text = document.parts[properid].text[int(start) - to_correct_off:int(start) - to_correct_off + len(row[2])]
                            if calc_ann_text != row[2]:
                                print(pmid, serial, paragraph, start, to_correct_off, id)

                            if entity_type == 'mutation':
                                ann = Annotation('e_2', int(start) - to_correct_off, row[2])
                                dataset.documents[pmid].parts[properid].annotations.append(ann)
                            elif entity_type == 'gene':
                                ann = Annotation('e_1', int(start) - to_correct_off, row[2])
                                dataset.documents[pmid].parts[properid].annotations.append(ann)

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
