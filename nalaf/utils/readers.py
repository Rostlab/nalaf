import abc
from bs4 import BeautifulSoup
import re
import glob
import csv
import os
import xml.etree.ElementTree as ET
import warnings

from nalaf.utils.download import DownloadArticle
from nalaf.structures.data import Dataset, Document, Part, Entity
from nalaf.utils.hdfs import maybe_get_hdfs_client, is_hdfs_directory, walk_hdfs_directory


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
        :returns: nalaf.structures.data.Dataset
        """
        return


class HTMLReader(Reader):
    """
    Reader for a local file system or hdfs of tagtog plain.html files.

    It reads either a single file or a directory (the contained .html's)
    """

    def __init__(self, path, whole_basename_as_docid=False, hdfs_url=None, hdfs_user=None):
        self.path = path
        """an html file or a directory containing .html files"""
        self.whole_basename_as_docid = whole_basename_as_docid

        self.hdfs_client = maybe_get_hdfs_client(hdfs_url, hdfs_user)

    def __read_directory_localfs(self):
        dataset = Dataset()

        filenames = glob.glob(str(self.path + "/**/*.html"), recursive=True) + glob.glob(str(self.path + "/**/*.xml"), recursive=True)
        for filename in filenames:
            dataset = self.__read_file_path_localfs(filename, dataset)

        return dataset

    def __read_file_path_localfs(self, filename, dataset=None):
        if dataset is None:
            dataset = Dataset()

        with open(filename, 'rb') as a_file:
            HTMLReader.read_file(a_file, filename, dataset, self.whole_basename_as_docid)

        return dataset

    def __read_directory_hdfs(self):
        dataset = Dataset()

        filenames = walk_hdfs_directory(self.hdfs_client, self.path, lambda fname: fname.endswith(".html") or fname.endswith(".xml"))
        for filename in filenames:
            dataset = self.__read_file_path_hdfs(filename, dataset)

        return dataset

    def __read_file_path_hdfs(self, filename, dataset=None):
        if dataset is None:
            dataset = Dataset()

        with self.hdfs_client.read(filename) as reader:
            HTMLReader.read_file(reader, filename, dataset, self.whole_basename_as_docid)

        return dataset

    @staticmethod
    def read_file(a_file, filename, dataset=None, whole_basename_as_docid=False):
        if dataset is None:
            dataset = Dataset()

        soup = BeautifulSoup(a_file, "html.parser")
        document = Document()

        for part in soup.find_all(id=re.compile('^s')):
            if re.match(r'^s[3-9]', part['id']):
                is_abstract = False
            else:
                is_abstract = True
            document.parts[part['id']] = Part(str(part.string), is_abstract=is_abstract)

        doc_id = os.path.basename(filename).replace('.plain.html', '').replace('.html', '').replace('.xml', '')
        if not whole_basename_as_docid and '-' in doc_id:
            doc_id = doc_id.split('-')[-1]

        dataset.documents[doc_id] = document

        return dataset

    def read(self):
        if self.hdfs_client is None:
            if os.path.isdir(self.path):
                return self.__read_directory_localfs()
            else:
                return self.__read_file_path_localfs(filename=self.path)

        else:
            if is_hdfs_directory(self.hdfs_client, self.path):
                return self.__read_directory_hdfs()
            else:
                return self.__read_file_path_hdfs(filename=self.path)

            return


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
        :returns: nalaf.structures.data.Dataset
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
        :returns: nalaf.structures.data.Dataset
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
        :returns: nalaf.structures.data.Dataset
        """
        dataset = Dataset()
        with DownloadArticle() as da:
            for pmid, doc in da.download(self.pmids):
                dataset.documents[pmid] = doc
        return dataset


class MedlineReader(Reader):
    """
    Reads in Medline citation sets  and creates a dataset with one document per medline citation.
    """
    def __init__(self, path):
        self.path = path
        """path to a single file or a directory containing medline.*.xml files"""

    def read(self):
        """
        :returns: nalaf.structures.data.Dataset
        """
        xmls = []
        if os.path.isdir(self.path):
            xmls = [os.path.join(root, file) for root, _, files in os.walk(self.path) for file in files
                    if file.startswith('medline') and file.endswith('xml')]
        elif self.path.startswith('medline') and self.path.endswith('xml'):
            xmls = [self.path]

        dataset = Dataset()

        for xml in xmls:
            for child in ET.parse(xml).getroot():
                pmid = next(child.iter('PMID')).text

                document = Document()
                article = next(child.iter('Article'))
                title = next(article.iter('ArticleTitle')).text
                document.parts['title'] = Part(title, is_abstract=False)
                try:
                    abstract = next(article.iter('AbstractText')).text
                    document.parts['abstract'] = Part(abstract)
                except StopIteration:
                    pass
                dataset.documents[pmid] = document

        return dataset


# TODO all following readers are deprecated and should be moved to nala


class SETHReader(Reader):
    """
    Reader for the SETH-corpus (http://rockt.github.io/SETH/)
    Format: PMID\tabstract (tab separated PMID and abstract)
    """

    def __init__(self, corpus_file):
        warnings.warn('This will be soon deleted and moved to _nala_', DeprecationWarning)
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
        warnings.warn('This will be soon deleted and moved to _nala_', DeprecationWarning)
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
                docid, title, abstract = row
                title = title.strip()
                abstract = abstract.strip()

                document = Document()
                if title:
                    document.parts['title'] = Part(title)
                if abstract and abstract != 'null':
                    document.parts['abstract'] = Part(abstract)

                dataset.documents[docid] = document

        return dataset


class VerspoorReader(Reader):
    """
    Reader for the Variome / Verspoor-corpus (http://www.opennicta.com.au/home/health/variome)
    Format: PMCID-serial-section-paragraph.txt: contains the text from a paragraph of the paper
    """

    def __init__(self, directory, mut_class_id, gene_class_id):
        import warnings
        warnings.warn('This will be soon deleted and moved to _nala_', DeprecationWarning)

        self.directory = directory
        """the directory containing the .html files"""
        self.mut_class_id = mut_class_id
        """
        class id that will be associated to the read (mutation) entities.
        """
        self.gene_class_id = gene_class_id
        """
        class id that will be associated to the read (gene / GGP) entities.
        """


    def read(self):
        """
        read each html file in the directory, parse it and create and instance of Document
        form a dataset consisting of every document parsed and return it

        Note that the text files may contain multiple paragraphs. The reader
        converts these paragraphs into different parts. Because of necessary offset corrections,
        the reader reads at the same time both the content and the annotations.

        :returns structures.data.Dataset
        """
        dataset = Dataset()

        file_list = glob.glob(str(self.directory + "/*.txt"))

        for file_path in file_list:
            file_name = os.path.basename(file_path)

            docid, partid_prefix, = file_name.replace('.txt', '').split('-', 1)
            # partid_prefix not complete due to multiple part cration for a single .txt file

            if 'Abstract' in partid_prefix:
                is_abstract = True
            else:
                is_abstract = False

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
                    partid = "{}-p{}".format(partid_prefix, i + 1)

                    if docid in dataset:
                        dataset.documents[docid].parts[partid] = Part(text_part, is_abstract=is_abstract)
                    else:
                        document = Document()
                        document.parts[partid] = Part(text_part, is_abstract=is_abstract)
                        dataset.documents[docid] = document

                    # add offset for next paragraph
                    tot_offset += len(text_part) + 2
                    offsets.append(tot_offset)

            # to delete last element
            del offsets[-1]

            # annotations
            with open(file_path.replace('.txt', '.ann'), encoding='utf-8') as f:
                reader = csv.reader(f, delimiter='\t')
                for row in reader:
                    if row[0].startswith('T'):
                        entity_type, start, end = row[1].split()
                        start = int(start)
                        end = int(end)
                        text = row[2]

                        partid = None
                        part_index = None

                        for i in range(len(offsets) - 1):
                            if offsets[i+1] > start:
                                part_index = i
                                break

                        if part_index is None:
                            part_index = len(offsets) - 1

                        partid = "{}-p{}".format(partid_prefix, part_index + 1)
                        real_start = start - offsets[part_index]
                        real_end = end - offsets[part_index]
                        calc_ann_text = document.parts[partid].text[real_start : real_end]

                        if calc_ann_text != text:
                            print("   ERROR", docid, part_index, partid, start, offsets, real_start, "\n\t", text, "\n\t", calc_ann_text, "\n\t", document.parts[partid].text)

                        if entity_type == 'mutation':
                            ann = Entity(self.mut_class_id, real_start, text)
                            dataset.documents[docid].parts[partid].annotations.append(ann)

                        elif entity_type == 'gene':
                            ann = Entity(self.gene_class_id, real_start, text)
                            dataset.documents[docid].parts[partid].annotations.append(ann)

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

    def __init__(self, corpus_file, mut_class_id):
        import warnings
        warnings.warn('This will be soon deleted and moved to _nala_', DeprecationWarning)

        self.corpus_file = corpus_file
        """the directory containing the .html files"""
        self.mut_class_id = mut_class_id
        """
        class id that will be associated to the read (mutation) entities.
        """

    def read(self):
        """
        :returns: nalaf.structures.data.Dataset
        """
        dataset = Dataset()

        with open(self.corpus_file, encoding='utf-8') as file:
            documents = file.read().strip().split('\n\n')
            for document_text in documents:
                lines = document_text.strip().splitlines()

                first_line = re.search('(\d+)\|t\|(.*)', lines[0])
                doc_id = first_line.group(1)
                tmvar_title = first_line.group(2)
                tmvar_abstract = re.search('(\d+)\|a\|(.*)', lines[1]).group(2)

                document = Document()
                title = Part(tmvar_title)
                abstract = Part(tmvar_abstract)
                document.parts['title'] = title
                document.parts['abstract'] = abstract

                for line in lines[2:]:
                    _, start, end, _, _, _ = line.split('\t')
                    start = int(start)
                    end = int(end)

                    if 0 <= start < end <= len(tmvar_title):
                        part = title
                    else:
                        part = abstract
                        start -= len(tmvar_title) + 1
                        end -= len(tmvar_title) + 1

                    part.annotations.append(Entity(self.mut_class_id, start, part.text[start:end]))

                dataset.documents[doc_id] = document

        return dataset


class OSIRISReaderMachineLearningReady(Reader):
    """
    Reads in the OSIRIS corpus by using the Wordfreak format which is aimed at
    specific annotations for machine learning tasks and does not represent the same annotations as the XML-file.

    # TODO still WIP (some minor bugs)
    """
    def __init__(self, path, mut_class_id):
        warnings.warn('This will be soon deleted and moved to _nala_', DeprecationWarning)
        self.path = os.path.abspath(path)
        """path to a folder containing files"""
        self.mut_class_id = mut_class_id
        """
        class id that will be associated to the read (mutation) entities.
        """

    def read(self):
        """
        :returns: nalaf.structures.data.Dataset
        """
        dataset = Dataset()
        for filename in glob.glob(self.path + '/*.txt'):
            with open(filename, 'r') as f:
                data = f.read()

                content = data.split("\n")
                try:
                    pmid = int(content[0])
                except ValueError:
                    continue

                doc = Document()

                title = content[2]
                part_title = Part(title, is_abstract=True)
                body = content[4]
                part_abstract = Part(body, is_abstract=True)

                title_offset = len(str(pmid)) + 2  # +2 for twice newline
                body_offset = title_offset + len(title) + 2  # +2 for twice newline

                # elements for temporary
                current_annotation = []
                last_element = None

                # print(filename, pmid, title)
                with open(filename + '.ann', 'r') as fa:
                    tree = ET.parse(fa)
                    for element in tree.iterfind('Annotation/Annotation[@type]'):
                        # if gene annotation skip
                        if element.attrib['type'] == 'ge':
                            continue

                        # if last element is empty (beginning of new doc) save as last_element and skip
                        if last_element is None:
                            last_element = element
                            continue

                        span = last_element.attrib['span'].split('..')
                        start = int(span[0])
                        end = int(span[1])
                        text = data[start:end]

                        if start >= body_offset:
                            norm_start = start - body_offset
                            norm_end = end - body_offset
                        else:
                            norm_start = start - title_offset
                            norm_end = end - title_offset

                        if end + 1 == int(element.attrib['span'].split('..')[0]):  # todo bugfix still mistake if space is in between the whole annotation case: "#1632 T"
                            if len(current_annotation) == 0:  # if no series of annotations linked
                                current_annotation.append(norm_start)
                                current_annotation.append(norm_end)
                                current_annotation.append(text)
                                current_annotation.append((start >= body_offset))  # if is_body
                            else:  # if already annotations contained there
                                current_annotation[1] = norm_end
                                current_annotation[2] += text
                        else:
                            if len(current_annotation) > 0:
                                entity = Entity(self.mut_class_id, current_annotation[0], current_annotation[2])
                                if current_annotation[3]:
                                    part_abstract.annotations.append(entity)
                                else:
                                    part_title.annotations.append(entity)
                                current_annotation = []

                            entity = Entity(self.mut_class_id, norm_start, text)
                            if start >= body_offset:
                                part_abstract.annotations.append(entity)
                            else:
                                part_title.annotations.append(entity)

                        last_element = element

                    span = last_element.attrib['span'].split('..')
                    start = int(span[0])
                    end = int(span[1])
                    text = data[start:end]
                    if len(current_annotation) == 0:  # if no series of annotations linked
                        if start >= body_offset:
                            norm_start = start - body_offset
                            is_body = True
                        else:
                            norm_start = start - title_offset
                            is_body = False

                        entity = Entity(self.mut_class_id, norm_start, text)

                        if is_body:
                            part_abstract.annotations.append(entity)
                        else:
                            part_title.annotations.append(entity)

                    else:  # if already annotations contained there
                        current_annotation[2] += text
                        entity = Entity(self.mut_class_id, current_annotation[0], current_annotation[2])
                        if current_annotation[3]:
                            part_abstract.annotations.append(entity)
                        else:
                            part_title.annotations.append(entity)

                doc.parts['title'] = part_title
                doc.parts['abstract'] = part_abstract
                # print(part_title)
                # print(part_body)
                dataset.documents[pmid] = doc
                # print(doc)

        return dataset


class OSIRISReader(Reader):
    """
    Parses the OSIRIS corpus by using their supplied XML-file alone.
    """
    def __init__(self, path, mut_class_id):
        warnings.warn('This will be soon deleted and moved to _nala_', DeprecationWarning)
        self.path = os.path.abspath(path)
        """path to xml file"""
        self.mut_class_id = mut_class_id
        """
        class id that will be associated to the read (mutation) entities.
        """

    def read(self):
        """
        :returns: nalaf.structures.data.Dataset
        """
        dataset = Dataset()
        with open(self.path, 'r') as f:

            tree = ET.parse(f)
            # level document
            for element in tree.iterfind('Article'):
                doc = Document()

                # pmid <Pmid>
                pmid = element[0].text

                # title <Title>
                title = element[1].text
                if not title:
                    title = ""
                title_annotations = []
                for child in element[1]:
                    if child.tag == 'variant':
                        entity = Entity(self.mut_class_id, len(title), child.text)
                        title_annotations.append(entity)
                    # unforunately child.text or child.tail can be empty and return None, which cannot be written as ""
                    try:
                        title += child.text
                    except TypeError:
                        pass
                    try:
                        title += child.tail
                    except TypeError:
                        pass
                part_title = Part(title)
                part_title.annotations.extend(title_annotations)

                # body - abstract <Abstract>
                abstract = element[2].text
                if not abstract:
                    abstract = ""
                abstract_annotations = []
                for child in element[2]:
                    if child.tag == 'variant':
                        entity = Entity(self.mut_class_id, len(abstract), child.text)
                        abstract_annotations.append(entity)
                    # unforunately child.text or child.tail can be empty and return None, which cannot be written as ""
                    try:
                        abstract += child.text
                    except TypeError:
                        pass
                    try:
                        abstract += child.tail
                    except TypeError:
                        pass
                part_abstract = Part(abstract)
                part_abstract.annotations.extend(abstract_annotations)

                # save part to document
                doc.parts['title'] = part_title
                doc.parts['abstract'] = part_abstract
                dataset.documents[pmid] = doc  # save document to dataset
        return dataset


class ProteinResidueCorpusPartialReader(Reader):
    """
    Reader for the LEAP-FS / Protein Residue (Full Text) Corpus
    http://bionlp-corpora.sourceforge.net/proteinresidue/

    Note:
        The reader practically actually only reads annotations.
        The text of the referenced PMIDs are not directly given and users
        should be get that directly from PubMed. This is not trivial since
        we do not exactly know how the original authors parsed the texts.

        Since the annotations contain start&end offsets and the entities texts,
        for every entity we create a different part that so spans the whole entity.


    Format:
        9724744	Mutation	31035	31062	Asp	450	Ala	D450 is replaced by alanine
        9724744	Mutation	38528	38541	Asp	450	Ala	Asp-450 → Ala
        9724744	Mutation	38556	38564	Asp	483	Ala	D483A
        9724744	Mutation	38566	38571	Arg	487	Ala	R487A
        9724744	Mutation	38613	38625	Asp	483	Ala	Asp-483 →Ala
        9724744	Mutation	38630	38643	Arg	487	Ala	Arg-487 → Ala
        9724744	AminoacidResidue	30956	30960	Asp	450	NULL	D450
    """

    def __init__(self, corpus_file, mut_class_id, residue_class_id):
        import warnings
        warnings.warn('This will be soon deleted and moved to _nala_', DeprecationWarning)

        self.corpus_file = corpus_file
        """the directory containing the .html files"""
        self.mut_class_id = mut_class_id
        """class id that will be associated to the read mutation entities (Mutation)."""
        self.residue_class_id = residue_class_id
        """class id that will be associated to the read residue entities (AminoacidResidue)."""


    def read(self):
        """
        :returns: nalaf.structures.data.Dataset
        """
        dataset = Dataset()

        with open(self.corpus_file, encoding='utf-8') as file:

            for row in file:
                columns = row.split("\t")

                docid = columns[0]
                typ = columns[1]
                start = columns[2]
                end = columns[3]
                entity_text = columns[7]

                class_id = None
                if typ == 'Mutation':
                    class_id = self.mut_class_id
                elif typ == 'AminoacidResidue':
                    class_id = self.residue_class_id

                if class_id:
                    document = dataset.documents.get(docid, Document())

                    part = Part(entity_text)
                    document.parts[typ + '|' + start + '|' + end] = part

                    part.annotations.append(Entity(class_id, int(start), entity_text))

                    dataset.documents[docid] = document

        return dataset
