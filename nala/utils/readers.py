import abc
from bs4 import BeautifulSoup
from nala.bootstrapping.utils import DownloadArticle
from nala.structures.data import Dataset, Document, Part, Entity, Relation
import re
import glob
import csv
import os
import json
from nala.utils import MUT_CLASS_ID


class Reader:
    """
    Abstract class for reading in a dataset in some format.
    Subclasses that inherit this class should:
    * Be named [Name]Reader
    * Implement the abstract method read that returns an object of type Dataset
    """

    @abc.abstractmethod
    def read(self):
        """
        :returns: nala.structures.data.Dataset
        """
        return


class HTMLReader(Reader):
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
                soup = BeautifulSoup(file, 'lxml')
                document = Document()

                for part in soup.find_all(id=re.compile('^s')):
                    if re.match(r'^s[3-9]', part['id']):
                        is_abstract = False
                    else:
                        is_abstract = True
                    document.parts[part['id']] = Part(str(part.string), is_abstract=is_abstract)

                basename = os.path.basename(filename)
                if '-' in basename:
                    dataset.documents[filename.split('-')[-1].replace('.plain.html', '')] = document
                else:
                    dataset.documents[basename.replace('.html', '')] = document
        return dataset


class SETHReader(Reader):
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


class MutationFinderReader(Reader):
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


class VerspoorReader(Reader):
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

        # last pmid
        last_pmid = ""
        id_counter = 0
        ids_per_file_array = [1]

        for file_path in file_list:
            file_name = os.path.basename(file_path)

            pmid, serial, part_type, *_, paragraph, = file_name.replace('.txt', '').split('-')
            # serial = first section
            # paragrahp = second section
            # partid = kind of part e.g. Abstract
            # divided by newlines = 3rd param which is p## or h##
            # NOTE h## follows with p## so no seperate calculations

            if 'Abstract' in part_type:
                is_abstract = True
            else:
                is_abstract = False

            with open(file_path, encoding='utf-8') as file:
                text_raw = file.read()

            text = text_raw.replace('** IGNORE LINE **\n', '')
            paragraph_list = text.split('\n\n')

            # check for last document the same as now, otherwise restart counter for ids
            if pmid != last_pmid:
                id_counter = 0
                ids_per_file_array = [1]
                last_pmid = pmid
            else:
                ids_per_file_array.append(id_counter + 1)  # id_counter finishes with last id of last file so --> +1

            # inital offset for raw_text
            tot_offset = text_raw.count('** IGNORE LINE **\n') * 18
            offsets = [tot_offset]

            for i, text_part in enumerate(paragraph_list):
                # if text is empty (usually last text due to splitting of "\n\n")
                if text_part != "":
                    first_serial = "s" + serial
                    second_serial = "s" + paragraph.replace("p", "")

                    # simple_id counter starts with 0 and before each use in simple_id var increment 1
                    id_counter += 1

                    # OPTIONAL to activate but annotation reader has to modified as well
                    # header when 10 >= words in text_part
                    if len(text_part.split(" ")) < 10:
                        paragraph_id = "h" + "{:02d}".format(i + 1)
                        simple_id = "s1h{}".format(id_counter)
                    else:
                        paragraph_id = "p" + "{:02d}".format(i + 1)
                        simple_id = "s1p{}".format(id_counter)

                    partid = "{0}{1}{2}".format(first_serial, second_serial, paragraph_id)

                    if pmid in dataset:
                        dataset.documents[pmid].parts[simple_id + partid] = Part(text_part, is_abstract=is_abstract)
                    else:
                        document = Document()
                        document.parts[simple_id + partid] = Part(text_part, is_abstract=is_abstract)
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
                                # id_h = "h{:02d}".format(i)

                                to_correct_off = last_off

                                simple_id = "s1p{}".format(ids_per_file_array[-1] + i - 1)  # -1 since starting to count with 1
                                # simple_id_h = "s1h{}".format(ids_per_file_array[-1] + i - 1)
                                break
                            last_off = off

                        # if last element
                        if id == "None":
                            id = "p{:02d}".format(len(offsets))

                            simple_id = "s1p{}".format(ids_per_file_array[-1] + len(offsets) - 1)
                            # simple_id_h = "s1h{}".format(ids_per_file_array[-1] + len(offsets) - 1)

                            to_correct_off = offsets[-1]

                        properid = "{0}{1}{2}".format(first_serial, second_serial, id)
                        # properid_h = "{0}{1}{2}".format(first_serial, second_serial, id_h)

                        if id == "None":
                            print("None???", pmid, serial, paragraph, start, offsets)

                        # if (simple_id + properid) not in dataset.documents[pmid].parts or (simple_id.replace("p", "h") + properid.replace("p", "h")) not in dataset.documents[pmid].parts:
                        #     print("NoKEY???", dataset.documents[pmid].parts)
                            # print(json.dumps(document, indent=4))

                        # print(document.parts[properid].text[int(start) - to_correct_off:int(start) - to_correct_off + len(row[2])], "==", row[2])

                        try:
                            calc_ann_text = document.parts[simple_id + properid].text[int(start) - to_correct_off:int(start) - to_correct_off + len(row[2])]
                        except KeyError:
                            simple_id = simple_id.replace("p", "h")
                            properid = properid.replace("p", "h")
                            calc_ann_text = document.parts[simple_id + properid].text[int(start) - to_correct_off:int(start) - to_correct_off + len(row[2])]


                        if calc_ann_text != row[2]:
                            print(pmid, serial, paragraph, start, to_correct_off, id)

                        if entity_type == 'mutation':
                            ann = Entity(MUT_CLASS_ID, int(start) - to_correct_off, row[2])
                            # try:
                            dataset.documents[pmid].parts[simple_id + properid].annotations.append(ann)
                            # except KeyError:
                            #     dataset.documents[pmid].parts[simple_id_h + properid_h].annotations.append(ann)

                        elif entity_type == 'gene':
                            ann = Entity('e_1', int(start) - to_correct_off, row[2])
                            # try:
                            dataset.documents[pmid].parts[simple_id + properid].annotations.append(ann)
                            # except KeyError:
                            #     dataset.documents[pmid].parts[simple_id_h + properid_h].annotations.append(ann)

        return dataset


class TmVarReader(Reader):
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
        """
        :returns: nala.structures.data.Dataset
        """
        dataset = Dataset()
        with open(self.corpus_file, encoding='utf-8') as file:

            for line in file:
                title = line.split('|t|')[-1]
                splitted = next(file).split(('|a|'))

                pmid = splitted[0]
                abstract = "".join([xs + " " for xs in splitted[1:]]).strip()

                document = Document()
                document.parts['abstract'] = Part(title + abstract)

                line = next(file)
                while line != '\n' and line != '':
                    _, start, end, text, *_ = line.split('\t')
                    document.parts['abstract'].annotations.append(Entity(MUT_CLASS_ID, int(start), text))
                    try:
                        oldline = line
                        line = next(file)
                    except StopIteration:
                        print('StopIteration Error, Line before:', oldline)
                        line = '\n'

                dataset.documents[pmid] = document

        return dataset


class StringReader(Reader):
    """
    Reads in a simple string and creates a dataset out of it.
    Useful for quick testing of model capabilities.

    The dataset contains a single Document with a single Part
    which contains the text given by the string.
    """

    def __init__(self, string):
        self.string = string
        """the string from which we form the dataset"""

    def read(self):
        """
        :returns: nala.structures.data.Dataset
        """
        part = Part(self.string)
        document = Document()
        dataset = Dataset()

        dataset.documents['doc_1'] = document
        document.parts['part_1'] = part

        return dataset


class TextFilesReader(Reader):
    """
    Reads in one or more file and creates a dataset out of them.
        * If the input is a path to a single file we create a dataset with one document
        * If the input is a path to a directory we create a dataset where each file is one document
    The name of the file is the document ID.
    Reads in only files with .txt extension.
    When reading in each file we separate into parts by blank lines if there are any.
    """

    def __init__(self, path):
        self.path = path
        """path to a single file or a directory containing input files"""

    @staticmethod
    def __process_file(filename):
        document = Document()
        with open(filename) as file:
            part_id = 1
            for part in re.split('\n\n', file.read()):
                if part.strip():
                    document.parts['{}'.format(part_id)] = Part(part)
                    part_id += 1

        return os.path.split(filename)[-1], document

    def read(self):
        """
        :returns: nala.structures.data.Dataset
        """
        dataset = Dataset()
        if os.path.isdir(self.path):
            for filename in glob.glob(self.path + '/*.txt'):
                doc_id, doc = self.__process_file(filename)
                dataset.documents[doc_id] = doc
        else:
            if os.path.splitext(self.path)[-1] == '.txt':
                doc_id, doc = self.__process_file(self.path)
                dataset.documents[doc_id] = doc
            else:
                raise Exception('not a .txt file extension')

        return dataset


class PMIDReader(Reader):
    """
    Reads in a single PMID or a list of PMIDs,
    downloads the associated articles (including title and abstract)
    and creates a dataset with one document per PMID.
    """
    def __init__(self, pmids):
        if hasattr(pmids, '__iter__'):
            self.pmids = pmids
        else:
            self.pmids = [pmids]

    def read(self):
        """
        :returns: nala.structures.data.Dataset
        """
        dataset = Dataset()
        with DownloadArticle(one_part=True) as da:
            for pmid, doc in da.download(self.pmids):
                dataset.documents[pmid] = doc
        return dataset
