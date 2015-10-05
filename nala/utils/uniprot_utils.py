from itertools import chain
import requests
from nala.utils.cache import Cacheable

class Uniprot(Cacheable):
    """
    Helper class that accesses the database identifier mapping service from Uniprot.
    and returns a list of 2-tuple with the requested geneid and the corresponding Uniprot ID.
    """

    def __init__(self):
        super().__init__()
        self.url = 'http://www.uniprot.org/mapping/'

    def get_uniprotid_for_entrez_geneid(self, *list_geneids):
        return_dict = {}
        to_be_downloaded = []

        for geneid in list_geneids:
            geneid = str(geneid)
            if geneid in self.cache:
                return_dict[geneid] = self.cache[geneid]
            else:
                to_be_downloaded.append(geneid)

        if len(to_be_downloaded) == 0:
            return return_dict

        params = {
            'from': 'P_ENTREZGENEID',
            'to': 'ACC',
            'format': 'tab',
            'query': ' '.join([str(x) for x in to_be_downloaded])
        }

        r = requests.get(self.url, params)
        results = r.text.splitlines()[1:]
        if not results:
            return return_dict

        found_genes = {}

        for line in results:
            geneid, uniprotid = line.split('\t')
            if geneid in found_genes:
                found_genes[geneid].append(uniprotid)
            else:
                found_genes[geneid] = [uniprotid]

        for key, value in found_genes.items():
            self.cache[key] = value

        return return_dict.update(found_genes)
