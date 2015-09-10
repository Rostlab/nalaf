import requests


class GNormPlus:
    """
    Helper class that accesses the rest API for GNormPlus from NCBI
    and returns a list of annotated genes for a given PMID
    """
    def __init__(self):
        self.url = 'http://www.ncbi.nlm.nih.gov/CBBresearch/Lu/Demo/RESTful/tmTool.cgi/Gene/{}/PubTator/'

    def get_genes_for_pmid(self, pmid):
        req = requests.get(self.url.format(pmid))

        genes = []
        for line in req.text.splitlines()[2:-1]:  # skip title and abstract
            _, start, end, text, _, gene_id = line.split('\t')
            genes.append((start, end, text, gene_id))
        return genes
