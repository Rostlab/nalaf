import requests
from nala.utils.cache import Cacheable


class GNormPlus(Cacheable):
    """
    Helper class that accesses the rest API for GNormPlus from NCBI
    and returns a list of annotated genes for a given PMID
    """

    def __init__(self):
        super().__init__()
        self.url = 'http://www.ncbi.nlm.nih.gov/CBBresearch/Lu/Demo/RESTful/tmTool.cgi/Gene/{}/PubTator/'

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
