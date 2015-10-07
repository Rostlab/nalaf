import abc
import re
from nala.utils.ncbi_utils import GNormPlus
from nala.utils.uniprot_utils import Uniprot
from nala.structures.data import Annotation, Relation
from nala.utils import MUT_CLASS_ID, PRO_CLASS_ID
from nala.utils import PRO_REL_MUT_CLASS_ID


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

    # todo change normalizazion_database to normalise option
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

    def __init__(self):
        super().__init__(['Gene', 'Protein'])

    def tag(self, dataset, annotated=True, uniprot=False):
        """
        :type dataset: nala.structures.data.Dataset
        :param annotated: if True then saved into annotations otherwise into predicted_annotations
        """
        with GNormPlus() as gnorm:
            for docid, doc in dataset.documents.items():
                if 'Conclusion' in doc.get_text():  # todo check whether this is enough for finding out if full document or not
                    genes = gnorm.get_genes_for_text(doc, docid, postproc=True)
                else:
                    genes = gnorm.get_genes_for_pmid(docid, postproc=True)

                # genes
                # if uniprot normalisation as well then:
                genes_mapping = {}
                if uniprot:
                    with Uniprot() as uprot:
                        list_of_ids = gnorm.uniquify_genes(genes)
                        genes_mapping = uprot.get_uniprotid_for_entrez_geneid(list_of_ids)
                last_index = -1
                part_index = 0
                for partid, part in doc.parts.items():
                    last_index = part_index
                    part_index += part.get_size() + 1
                    for gene in genes:
                        if gene[2] in part.text and last_index < gene[0] < part_index:
                            start = gene[0] - last_index
                            # todo discussion which confidence value for gnormplus because there is no value supplied
                            ann = Annotation(class_id=PRO_CLASS_ID, offset=start, text=gene[2], confidence=0.5)
                            try:
                                norm_dict = {
                                    'EntrezGeneID': gene[3],
                                    'UniprotID': genes_mapping[gene[3]]
                                }
                            except KeyError:
                                norm_dict = {'EntrezGeneID': gene[3]}

                            norm_string = ''  # todo normalized_text (stemming ... ?)
                            ann.normalisation_dict = norm_dict
                            ann.normalized_text = norm_string
                            if annotated:
                                part.annotations.append(ann)
                            else:
                                part.predicted_annotations.append(ann)


class RelationshipExtractionGeneMutation(Tagger):
    def __init__(self):
        super().__init__(['GeneRelMutation', 'ProteinRelMutation'])

    def tag(self, dataset):
        from itertools import product
        for part in dataset.parts():
            for ann_1, ann_2 in product(
                    (ann for ann in part.predicted_annotations if ann.class_id == MUT_CLASS_ID),
                    (ann for ann in part.predicted_annotations if ann.class_id == PRO_CLASS_ID)):
                if part.get_sentence_index_for_annotation(ann_1) == part.get_sentence_index_for_annotation(ann_2):
                    part.relations.append(
                        Relation(ann_1.offset, ann_2.offset, ann_1.text, ann_2.text, PRO_REL_MUT_CLASS_ID))