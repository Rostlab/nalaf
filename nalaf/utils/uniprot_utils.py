from itertools import chain
import requests
from nalaf.utils.cache import Cacheable

class Uniprot(Cacheable):
    """
    Helper class that accesses the database identifier mapping service from Uniprot.
    and returns a list of 2-tuple with the requested geneid and the corresponding Uniprot ID.
    """

    def __init__(self):
        super().__init__()
        self.url = 'http://www.uniprot.org/mapping/'

    def get_uniprotid_for_entrez_geneid(self, list_geneids):
        """
        Get dictionary mapping from { EntrezGeneID : [ UniprotID, ... ]
        :param list_geneids:
        :type list_geneids: [int] or [str] or int or str
        :return: dictionary geneid --> uniprotid-list
        """
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

        # todo test this part again
        for line in results:
            geneid, uniprotid = line.split('\t')
            if geneid in return_dict:
                return_dict[geneid].append(uniprotid)
                self.cache[geneid].append(uniprotid)
            else:
                return_dict[geneid] = [uniprotid]
                self.cache[geneid] = [uniprotid]

        return return_dict
