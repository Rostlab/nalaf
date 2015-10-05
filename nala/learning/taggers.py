import abc
import re
from utils.ncbi_utils import GNormPlus
from utils.uniprot_utils import Uniprot


class Tagger:
    """
    Abstract class for tagging a dataset with predicted annotations.

    Subclasses that inherit this class should:
    * Be named [Name]Tagger
    * Implement the abstract method tag
    * Use some sort of model or service to generate predictions
        * If you only want to read in predictions already saved in ann.json
         use AnnJsonAnnotationReader with _is_predicted = True
    * Append new items to the list field "predicted_annotations" of each Part in the dataset
    * Set the meta_attribute predicts_classes

    Optionally the implementation may perform normalization of the predicted entities.
    In that case:
    * set the meta attribute performs_normalization = True
    * set the meta attribute normalization_database
    * set the fields normalized_id and normalized text for each Annotation object you create

    :type performs_normalization: bool
    :type normalization_database: str
    :type predicts_classes: list[str]
    """

    def __init__(self, predicts_classes):
        self.performs_normalization = False
        """whether this tagger also performs normalization"""
        self.normalization_database = ''
        """additional info about the normalization database, e.g. URL"""
        self.predicts_classes = predicts_classes
        """a list of class IDs that this tagger can predict"""

    @abc.abstractmethod
    def tag(self, dataset):
        """
        :type dataset: nala.structures.data.Dataset
        """
        pass


class CRFSuiteMutationTagger(Tagger):
    """
    Performs tagging with a binary model using CRFSuite

    :type crf_suite: nala.learning.crfsuite.CRFSuite
    """

    def __init__(self, predicts_classes, crf_suite, model_file='default_model'):
        super().__init__(predicts_classes)
        self.crf_suite = crf_suite
        """an instance of CRFSuite used to actually generate predictions"""
        self.model_file = model_file
        """path to the binary model used for generating predictions"""

    def tag(self, dataset):
        """
        :type dataset: nala.structures.data.Dataset
        """
        self.crf_suite.create_input_file(dataset, 'predict')
        self.crf_suite.tag('-m {} -i predict > output.txt'.format(self.model_file))
        self.crf_suite.read_predictions(dataset)


class GNormPlusGeneTagger(Tagger):
    """
    Performs tagging for genes with GNormPlus.
    Is able to add normalisations for uniprot as well.

    :type crf_suite: nala.learning.crfsuite.CRFSuite
    """

    def __init__(self, predicts_classes, normalise_uniprot=False):
        super().__init__(predicts_classes)
        self.uniprot=normalise_uniprot

    def tag(self, dataset):
        """
        :type dataset: nala.structures.data.Dataset
        """
        with GNormPlus() as gnorm:
            for docid, doc in dataset.documents.items():
                if doc.get_text().contains('Conclusion'):
                    genes = gnorm.get_genes_for_text(doc, docid, postproc=True)
                else:
                    genes = gnorm.get_genes_for_pmid(docid, postproc=True)

                # genes
                # if uniprot normalisation as well then:
                genes_mapping = None
                if self.uniprot:
                    with Uniprot() as uprot:
                        list_of_ids = gnorm.uniquify_genes(genes)
                        genes_mapping = uprot.get_uniprotid_for_entrez_geneid(list_of_ids)

                part_index = 0
                last_part_index = -1
