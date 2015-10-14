from itertools import chain
from xml.etree import ElementTree as ET
import requests
from nala.structures.data import Document, Part
from nala.utils.cache import Cacheable

__author__ = 'Aleksandar'


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

    def __init__(self, one_part=False):
        super().__init__()
        self.one_part = one_part
        """whether to put everything (title, abstract, etc.) under the same part joined with new line"""
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

            if self.one_part:
                joined_text = '\n'.join(element.text for element in
                                        chain(xml.findall('.//ArticleTitle'), xml.findall('.//AbstractText')))
                doc.parts['title_and_abstract'] = Part(joined_text)
            else:
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
    from nala.structures.data import Dataset
    from nala.structures.selection_pipelines import DocumentSelectorPipeline
    from itertools import count
    c = count(1)

    dataset = Dataset()
    with DocumentSelectorPipeline() as dsp:
        for pmid, document in dsp.execute():
            dataset.documents[pmid] = document
            # if we have generated enough documents stop
            if next(c) == n:
                break

    return dataset