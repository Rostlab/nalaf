import requests
import xml.etree.ElementTree as ET


class FilterByKeywords:
    def __init__(self):
        self.pubmed_url = 'http://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi'
        self.keywords = ('mutation', 'variation', 'substitution', 'insertion', 'deletion', 'snp')

    def filter(self, pubmed_ids):
        for pubmed_id in pubmed_ids:
            req = requests.get(self.pubmed_url, {'db': 'pubmed', 'retmode': 'xml', 'id': pubmed_id})
            xml = ET.fromstring(req.content)

            if any((
                    any(any(keyword in elem.text.lower() for keyword in self.keywords)
                        for elem in xml.findall('.//ArticleTitle')),
                    any(any(keyword in elem.text.lower() for keyword in self.keywords)
                        for elem in xml.findall('.//AbstractText')))):
                yield pubmed_id


class SwissProtDocumentSelector:
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
