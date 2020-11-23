import abc
import json
import requests
import os
from nalaf.structures.data import Dataset, Document, Part, Entity


# todo major refactor to learning/taggers class
class Tagger():
    """
    Abstract class for external tagger, like tmVar.
    """
    @abc.abstractmethod
    def generate(self, dataset):
        raise Exception('DEPRECATED -- This shimply should not be here. Use Annotator definition in annotators.py')
        # import warnings
        # warnings.warn(...)

        """
        Generates annotations from an external method, like tmVar or SETH.
        :type nalaf.structures.Dataset:
        :return: new dataset with annotations
        """
        # get annotated documents from somewhere
        return dataset


class TmVarTagger(Tagger):
    """
    TmVar tagger using the RESTApi from "http://www.ncbi.nlm.nih.gov/CBBresearch/Lu/Demo/tmTools/".
    """

    def __init__(self, mut_class_id):
        import warnings
        warnings.warn('This will be soon deleted and moved to _nala_', DeprecationWarning)

        self.mut_class_id = mut_class_id
        """
        class id that will be associated to the read (mutation) entities.
        """


    def generate(self, dataset):
        """
        :param dataset: TODO
        :return:
        """
        # todo docset
        # todo textfile tagger @major
        # generate pubtator object using PubtatorWriter
        # _tmp_pubtator_send = "temp_pubtator_file.txt"

        # submit to tmtools

        # receive new pubtator object

        # parse to dataset object using TmVarReader


    def generate_abstracts(self, list_of_pmids):
        """
        Generates list of documents using pmids and the restapi interface from tmtools.
        Source: "http://www.ncbi.nlm.nih.gov/CBBresearch/Lu/Demo/tmTools/"
        :param list_of_pmids: strings
        :return nalaf.structures.Dataset: dataset
        """
        # if os.path.isfile('cache.json'):
        #     with open('cache.json') as f:
        #           tm_var = json.load()
        # else:
        url_tmvar = 'http://www.ncbi.nlm.nih.gov/CBBresearch/Lu/Demo/RESTful/tmTool.cgi/Mutation/{0}/JSON/'
        # url_converter = 'http://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/'

        # load cache.json if exists
        if os.path.exists('cache.json'):
            with open('cache.json', 'r', encoding='utf-8') as f:
                tm_var = json.load(f)
        else:
            tm_var = {}

        for pmid in list_of_pmids:
            if pmid not in tm_var:  # if pmid was not already downloaded from tmTools
                req = requests.get(url_tmvar.format(pmid))
                try:
                    tm_var[pmid] = req.json()
                except ValueError:
                    pass
        # cache the tmVar annotations so we don't pull them every time
        with open('cache.json', 'w') as file:
            json.dump(tm_var, file, indent=4)

        # for key in tm_var:
        #     print(json.dumps(tm_var[key], indent=4))

        dataset = Dataset()
        for doc_id in list_of_pmids:
            if doc_id in tm_var:
                doc = Document()
                text = tm_var[doc_id]['text']
                part = Part(text)
                denotations = tm_var[doc_id]['denotations']
                annotations = []
                for deno in denotations:
                    ann = Entity(class_id=self.mut_class_id, offset=int(deno['span']['begin']), text=text[deno['span']['begin']:deno['span']['end']])
                    annotations.append(ann)
                    # note should the annotations from tmvar go to predicted_annotations or annotations?
                part.annotations = annotations
                doc.parts['abstract'] = part
                dataset.documents[doc_id] = doc

        return dataset
