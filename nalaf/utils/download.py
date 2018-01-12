from itertools import chain

import requests

from nalaf.structures.data import Document, Part
from nalaf.utils.cache import Cacheable
from xml.etree import ElementTree as ET


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
        self.is_timed = False

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
                # for now only include title and abstract
                title_elem = xml.find('.//ArticleTitle')
                if title_elem is not None:
                    doc.parts['title'] = Part(title_elem.text)

                abstract_elem = xml.findall('.//AbstractText')
                if abstract_elem is not None:
                    abstract_elems = []
                    for elem in abstract_elem:
                        if 'Label' in elem.attrib and elem.attrib['Label'] != 'UNLABELLED':
                            abstract_elems.append('{}: {}'.format(elem.attrib['Label'], elem.text))
                        else:
                            abstract_elems.append(elem.text)

                    abstract_elems = filter(None, abstract_elems)

                    doc.parts['abstract'] = Part(' '.join(abstract_elems))

            # yield the document but only if you found anything
            if len(doc.parts) > 0:
                yield pmid, doc
