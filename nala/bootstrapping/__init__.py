import glob
import os
import re
import requests
import xml.etree.ElementTree as ET
from nala.learning.crfsuite import CRFSuite
from nala.structures.data import Dataset, Document, Part
from nala.utils.cache import Cacheable
from nala.utils.readers import HTMLReader
from nala.utils.annotation_readers import AnnJsonAnnotationReader
from nala.utils.writers import TagTogFormat


class UniprotDocumentSelector(Cacheable):
    """
    Selects a list of pubmed IDs (articles) that are likely to have mutation mentions.

    Outline of the selection procedure:
    1. Select proteins given a uniprot query (by default Swiss-Prot human proteins)
    2. For each protein search for articles showing evidence of sequence variant of mutagenesis
    3. Return pubmed IDs (articles) associated with the evidence
    """

    def __init__(self):
        super().__init__()
        self.processed = set()
        self.uniprot_url = 'http://www.uniprot.org/uniprot/'

    def _get_uniprot_ids(self, query=None):
        if not query:
            query = '(annotation:(type:natural_variations) OR annotation:(type:mutagen))' \
                    ' AND reviewed:yes AND organism:"Homo sapiens (Human) [9606]"'
        params = {'query': query,
                  'columns': 'id',
                  'format': 'tab'}

        if query in self.cache:
            lines = self.cache[query]
        else:
            req = requests.get(self.uniprot_url, params)
            lines = req.text.splitlines()
            self.cache[query] = lines

        for uniprot_id in lines[1:]:  # skip first line
            yield uniprot_id

    def _get_pubmed_ids_for_protein(self, uniprot_id):
        if uniprot_id in self.cache:
            text = self.cache[uniprot_id]
        else:
            req = requests.get(self.uniprot_url + '{}.xml'.format(uniprot_id))
            text = req.text
            self.cache[uniprot_id] = text

        xml = ET.fromstring(text)
        ns = {'u': 'http://uniprot.org/uniprot'}  # namespace

        evidence_ids = []
        for elem in xml.findall('.//u:feature[@evidence]', ns):
            if elem.attrib['type'] in ('sequence variant', 'mutagenesis site'):
                evidence_ids.append(elem.attrib['evidence'])

        for elem in xml.findall('.//u:evidence[@key="{}"]/u:source/u:dbReference[@type="PubMed"]'.format(18), ns):
            pubmed_id = elem.attrib['id']
            # check to see if we have seen this id before
            if pubmed_id not in self.processed:
                self.processed.add(pubmed_id)
                yield pubmed_id

    def get_pubmed_ids(self):
        for uniprot_id in self._get_uniprot_ids():
            for pubmed_id in self._get_pubmed_ids_for_protein(uniprot_id):
                yield pubmed_id


class DownloadArticle(Cacheable):
    """
    A utility generator that for a given iterable of PMIDs generates Document objects
    created by downloading the articles associated with the pmid.
    """

    def __init__(self):
        super().__init__()
        self.pubmed_url = 'http://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi'

    def download(self, pmids):
        for pmid in pmids:
            if pmid in self.cache:
                xml = ET.fromstring(self.cache[pmid])
            else:
                req = requests.get(self.pubmed_url, {'db': 'pubmed', 'retmode': 'xml', 'id': pmid})
                text = req.text
                xml = ET.fromstring(text)
                self.cache[pmid] = text

            doc = Document()
            counter = 0

            # for now only include title and abstract
            for elem in xml.findall('.//ArticleTitle'):
                doc.parts['part_{}'.format(counter)] = Part(elem.text)
                counter += 1
            for elem in xml.findall('.//AbstractText'):
                doc.parts['part_{}'.format(counter)] = Part(elem.text)
                counter += 1

            # yield the document but only if you found anything
            if len(doc.parts) > 0:
                yield pmid, doc


def generate_documents(n):
    """
    Generate a given number of documents for bootstrapping applying the default filters.

    :param n: how many documents do you want
    :type n: int
    :returns: nala.structures.data.Dataset
    """
    from nala.bootstrapping import UniprotDocumentSelector
    from nala.bootstrapping.document_filters import KeywordsDocumentFilter
    from nala.bootstrapping.pmid_filters import AlreadyConsideredPMIDFilter
    from nala.bootstrapping import DownloadArticle
    from itertools import count
    c = count(1)

    dataset = Dataset()

    # use in context manager to enable caching
    with UniprotDocumentSelector() as uds:
        with DownloadArticle() as da:
            for pmid, document in \
                    KeywordsDocumentFilter().filter(
                        da.download(
                            AlreadyConsideredPMIDFilter(r'C:\Users\Aleksandar\Desktop\root', 4).filter(
                                uds.get_pubmed_ids()))):
                dataset.documents[pmid] = document
            # if we have generated enough documents stop
                if next(c) == n:
                    break

    return dataset


class Iteration(Cacheable):
    def __init__(self, folder=None):
        super().__init__()

        if folder is not None:
            self.bootstrapping_folder = folder
        else:
            self.bootstrapping_folder = "resources/bootstrapping"

        # represents the iteration
        self.number = -1

        # todo discussion on config file in bootstrapping root or iteration_n check for n

        # find iteration number
        _iteration_name = self.bootstrapping_folder + "/iteration_*/"
        for fn in glob.glob(_iteration_name):
            match = re.search(r'/iteration_(\d+)/$', fn)
            found_iteration = int(match.group(1))
            if found_iteration > self.number:
                self.number = found_iteration

    def learning(self):
        # parse base + reviewed files
        learning_data = HTMLReader(self.bootstrapping_folder + "iteration_")

        crf = CRFSuite("crfsuite")  # todo check for ability to use crfsuite command instead of abspath
        pass
        # todo learning
        # learning

        # docselection

        # tagging

        # manual review

        # automatic evaluation