import requests
import json
import os


class GNormPlus:
    """
    Helper class that accesses the rest API for GNormPlus from NCBI
    and returns a list of annotated genes for a given PMID
    """
    def __init__(self):
        self.url = 'http://www.ncbi.nlm.nih.gov/CBBresearch/Lu/Demo/RESTful/tmTool.cgi/Gene/{}/PubTator/'
        if os.path.exists('gnp_cache.json'):
            self.cache = json.load(open('gnp_cache.json'))
        else:
            self.cache = {}

    def __del__(self):
        if self.cache:
            with open('gnp_cache.json', 'w') as file:
                json.dump(self.cache, file)

    def get_genes_for_pmid(self, pmid):
        if pmid in self.cache:
            text = self.cache[pmid]
        else:
            req = requests.get(self.url.format(pmid))
            text = req.text
            self.cache[pmid] = text

        genes = []
        for line in text.splitlines()[2:-1]:  # skip title and abstract
            _, start, end, text, _, gene_id = line.split('\t')
            genes.append((int(start), int(end), text, gene_id))
        return genes
