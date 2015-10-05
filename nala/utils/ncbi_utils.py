import requests
import time
from nala.utils.cache import Cacheable


class GNormPlus(Cacheable):
    """
    Helper class that accesses the rest API for GNormPlus from NCBI
    and returns a list of annotated genes for a given PMID
    """

    def __init__(self):
        super().__init__()
        self.url = 'http://www.ncbi.nlm.nih.gov/CBBresearch/Lu/Demo/RESTful/tmTool.cgi/Gene/{}/PubTator/'
        self.baseurl = "http://www.ncbi.nlm.nih.gov/CBBresearch/Lu/Demo/RESTful/tmTool.cgi/"

    def get_genes_for_pmid(self, pmid):
        if pmid in self.cache:
            text = self.cache[pmid]
        else:
            req = requests.get(self.url.format(pmid))
            text = req.text
            self.cache[pmid] = text

        genes = []
        for line in text.splitlines()[2:-1]:  # skip title and abstract
            try:
                _, start, end, text, _, gene_id = line.split('\t')
                genes.append((int(start), int(end), text, gene_id))
            # the provided pmid was not a valid one
            except ValueError:
                pass
        return genes

    def get_genes_for_text(self, doc):
        """
        Retrieval via RESTful API with full documents.
        Attention!: one call can take a very long time. (no idea why, but sometimes it takes years and might not even finish)
        :param doc: Document that is supplied
        :type doc: nala.structures.data.Document
        :return: list of GeneIDs in EntrezGene-Format (Number)
        """
        title = doc.get_title()
        if title in self.cache:
            text = self.cache[title]
        else:
            data = "sampleid|t|" + doc.get_title() + '\n' + "sampleid|a|" + doc.get_body() + '\n'
            req = requests.post(self.baseurl + 'GNormPlus/Submit/', data=data)
            id = req.text
            status = 'Not yet'
            while status.startswith('Not yet'):
                req = requests.get(self.baseurl + id + '/Receive/')
                status = req.text
                time.sleep(5)
            print(status)
            # todo clean print statement
            self.cache[title] = status

        genes = []
        for line in text.splitlines()[2:-1]:  # skip title and abstract
            try:
                _, start, end, text, _, gene_id = line.split('\t')
                genes.append((int(start), int(end), text, gene_id))
            # the provided pmid was not a valid one
            except ValueError:
                pass
        return genes

