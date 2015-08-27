from itertools import chain
import json
import os
import requests
from utils import MUT_CLASS_ID


class NLMentionSelector:
    """
    Abstract class for determining whether an annotation in the dataset is a natural language (NL) mention.
    Subclasses that inherit this class should:
    * Be named [Name]Selector
    * Implement the abstract method define
    * Set the value
    """

    @abc.abstractmethod
    def define(self, dataset):
        """
        :type dataset: nala.structures.data.Dataset
        """
        return


class TMVarSelector(NLMentionSelector):
    """
    Definer based on the complete tmVar NER pipeline.

    Algorithm:
    run tmVar on our dataset and obtain predictions for mutations

    if a mutation annotation is predicted by tmVar
        then it is considered a standard mention
    otherwise we consider it a natural language mention.

    NOTE: Make sure to clear the cache (delete cache.ini file) when running the definer for different corpora.

    Implements the abstract class NLDefiner.
    """

    def define(self, dataset):
        if os.path.isfile('cache.json'):
            tm_var = json.load(open('cache.json'))
        else:
            url_tmvar = 'http://www.ncbi.nlm.nih.gov/CBBresearch/Lu/Demo/RESTful/tmTool.cgi/Mutation/{0}/JSON/'
            url_converter = 'http://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/'

            tm_var = {}
            for pmid, doc in dataset.documents.items():
                # if we have a pmc id instead converted it first to pmid
                if pmid.startswith('PMC'):
                    req = requests.get(url_converter, {'ids': pmid, 'format': 'json'})
                    pmid = req.json()['records'][0]['pmid']

                req = requests.get(url_tmvar.format(pmid))
                try:
                    tm_var[pmid] = req.json()
                except ValueError:
                    pass
            # cache the tmVar annotations so we don't pull them every time
            with open('cache.json', 'w') as file:
                file.write(json.dumps(tm_var, indent=4))

        for doc_id, doc in dataset.documents.items():
            if doc_id in tm_var:
                denotations = tm_var[doc_id]['denotations']
                text = tm_var[doc_id]['text']
                denotations = [text[d['span']['begin']:d['span']['end']] for d in denotations]

                for part_id, part in doc.parts.items():
                    for ann in chain(part.annotations, part.predicted_annotations):
                        if ann.class_id == MUT_CLASS_ID and ann.text not in denotations:
                            ann.subclass = True
