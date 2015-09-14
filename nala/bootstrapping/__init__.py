import requests
import xml.etree.ElementTree as ET
from nala.structures.data import Dataset, Document, Part


class SwissProtDocumentSelector:
    # TODO Add docstring
    def __init__(self):
        self.processed = set()
        self.uniprot_url = 'http://www.uniprot.org/uniprot/'

    def get_uniprot_ids(self):
        params = {'query': '(annotation:(type:natural_variations) OR annotation:(type:mutagen))'
                           ' AND reviewed:yes AND organism:"Homo sapiens (Human) [9606]"',
                  'columns': 'id',
                  'format': 'tab'}

        req = requests.get(self.uniprot_url, params)
        line_iterator = req.iter_lines(decode_unicode=True)
        next(line_iterator)  # skip first line

        for uniprot_id in line_iterator:
            yield uniprot_id

    def get_pubmed_ids_for_protein(self, uniprot_id):
        req = requests.get(self.uniprot_url + '{}.xml'.format(uniprot_id))
        xml = ET.fromstring(req.content)
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
        for uniprot_id in self.get_uniprot_ids():
            for pubmed_id in self.get_pubmed_ids_for_protein(uniprot_id):
                yield pubmed_id


class DownloadArticle:
    """
    A utility generator that for a given iterable of PMIDs generates Document objects
    created by downloading the articles associated with the pmid.
    """
    def __init__(self):
        self.pubmed_url = 'http://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi'

    def download(self, pmids):
        for pmid in pmids:
            req = requests.get(self.pubmed_url, {'db': 'pubmed', 'retmode': 'xml', 'id': pmid})
            xml = ET.fromstring(req.content)

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
                yield doc


def generate_documents(n):
    """
    Generate a given number of documents for bootstrapping applying the default filters.

    :param n: how many documents do you want
    :type n: int
    """
    from nala.bootstrapping import SwissProtDocumentSelector
    from nala.bootstrapping.document_filters import KeywordsDocumentFilter
    from nala.bootstrapping.pmid_filters import AlreadyConsideredPMIDFilter
    from nala.bootstrapping import DownloadArticle
    from itertools import count
    c = count(1)

    for document in \
        KeywordsDocumentFilter().filter(
            DownloadArticle().download(
                AlreadyConsideredPMIDFilter(r'C:\Users\Aleksandar\Desktop\root', 4).filter(
                    SwissProtDocumentSelector().get_pubmed_ids()))):
        yield document

        # if we have generated enough documents stop
        if next(c) == n:
            break

if __name__ == '__main__':
    # example usage
    # TODO document this better and move it to a more appropriate place
    list(generate_documents(2))
