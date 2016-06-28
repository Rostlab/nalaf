import abc
import difflib
from nalaf.utils.ncbi_utils import GNormPlus
from nalaf.utils.uniprot_utils import Uniprot
from nalaf.structures.data import Entity, Relation
from nalaf.utils import MUT_CLASS_ID, PRO_CLASS_ID, PRO_REL_MUT_CLASS_ID, ENTREZ_GENE_ID, UNIPROT_ID


class Annotator:
    """
    Abstract class for Entity tagging or Relationship tagging.
    This forms a hierarchy, where Tagger and RelationExtractor are abstract
    subclasses of Annotator

    Any Named Entity Recognizer should inherit from the class Tagger, and
    should:
    * Be named [Name]Tagger
    * Implement the abstract method tag
    """
    def __init__(self, predicts_classes):
        self.predicts_classes = predicts_classes
        """a list of class IDs that this tagger can predict"""


class Tagger(Annotator):
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
    * set the meta attribute does_normalization = True
    * set the meta attribute normalization_database
    * set the fields normalized_id and normalized text for each Annotation object you create

    :type does_normalization: bool
    :type normalization_database: str
    :type predicts_classes: list[str]
    """

    # todo change normalizazion_database to normalise option
    def __init__(self, predicts_classes):
        super().__init__(predicts_classes)
        self.does_normalization = False
        """whether this tagger also performs normalization"""
        self.normalization_database = ''
        """additional info about the normalization database, e.g. URL"""

    @abc.abstractmethod
    def tag(self, dataset):
        """
        :type dataset: nalaf.structures.data.Dataset
        """
        pass


class RelationExtractor(Annotator):
    """
    Abstract class for tagging a dataset with predicted relations between
    entities.

    Subclasses that inherit this class should:
    * Be named [Name]RelationExtractor
    * Implement the abstract method annotate
    * Use some sort of model or service to generate predictions
        * If you only want to read in predictions already saved in ann.json
          use AnnJsonAnnotationReader with _is_predicted = True
        * This will not only read the entities, but also the relations
    * Append new items to the list field "predicted_relations" of each Part in the dataset
    * Set the meta_attribute predicts_classes

    :type does_normalization: bool
    :type normalization_database: str
    :type predicts_classes: list[str]
    """
    def __init__(self, entity1_class, entity2_class, relation_type):
        self.entity1_class = entity1_class
        """class id of the first entity"""
        self.entity2_class = entity2_class
        """class id of the second entity"""
        self.relation_type = relation_type
        """the type of relation between the two entiies in predicts_classes"""

    @abc.abstractmethod
    def tag(self, dataset):
        """
        :type dataset: nalaf.structures.data.Dataset
        """
        pass


class StubSameSentenceRelationExtractor(RelationExtractor):

    def __init__(self, entity1_class, entity2_class, relation_type):
        super().__init__(entity1_class, entity2_class, relation_type)

    def tag(self, dataset):
        from itertools import product
        for part in dataset.parts():
            for ann_1, ann_2 in product(
                    (ann for ann in part.predicted_annotations if ann.class_id == self.entity1_class),
                    (ann for ann in part.predicted_annotations if ann.class_id == self.entity2_class)):
                if part.get_sentence_index_for_annotation(ann_1) == part.get_sentence_index_for_annotation(ann_2):
                    part.predicted_relations.append(
                        Relation(ann_1.offset, ann_2.offset, ann_1.text, ann_2.text, self.relation_type))


class CRFSuiteTagger(Tagger):
    """
    Performs tagging with a binary model using CRFSuite

    :type crf_suite: nalaf.learning.crfsuite.CRFSuite
    """

    def __init__(self, predicts_classes, crf_suite, model_file='example_entity_model'):
        super().__init__(predicts_classes)
        self.crf_suite = crf_suite
        """an instance of CRFSuite used to actually generate predictions"""
        self.model_file = model_file
        """path to the binary model used for generating predictions"""

    def tag(self, dataset):
        """
        :type dataset: nalaf.structures.data.Dataset
        """
        self.crf_suite.create_input_file(dataset, 'predict')
        self.crf_suite.tag('-m {} -i predict > output.txt'.format(self.model_file))
        self.crf_suite.read_predictions(dataset)


class GNormPlusGeneTagger(Tagger):
    """
    Performs tagging for genes with GNormPlus.
    Is able to add normalisations for uniprot as well.

    :type crf_suite: nalaf.learning.crfsuite.CRFSuite
    """

    def __init__(self):
        super().__init__([ENTREZ_GENE_ID, UNIPROT_ID])

    def __find_offset_adjustments(self, s1, s2, start_offset):
        return [(start_offset+i1, j2-j1-i2+i1)
                   for type, i1, i2, j1, j2  in difflib.SequenceMatcher(None, s1, s2).get_opcodes()
                   if type in ['replace', 'insert']]

    def tag(self, dataset, annotated=False, uniprot=False, process_only_abstract=True):
        """
        :type dataset: nalaf.structures.data.Dataset
        :param annotated: if True then saved into annotations otherwise into predicted_annotations
        """
        with GNormPlus() as gnorm:
            for doc_id, doc in dataset.documents.items():
                if process_only_abstract:
                    genes, gnorm_title, gnorm_abstract = gnorm.get_genes_for_pmid(doc_id, postproc=True)

                    if uniprot:
                        with Uniprot() as uprot:
                            list_of_ids = gnorm.uniquify_genes(genes)
                            genes_mapping = uprot.get_uniprotid_for_entrez_geneid(list_of_ids)
                    else:
                        genes_mapping = {}

                    # find the title and the abstract
                    parts = iter(doc.parts.values())
                    title = next(parts)
                    abstract = next(parts)
                    adjustment_offsets = []
                    if title.text != gnorm_title:
                        adjustment_offsets += self.__find_offset_adjustments(title.text, gnorm_title, 0)
                    if abstract.text != gnorm_abstract:
                        adjustment_offsets += self.__find_offset_adjustments(abstract.text, gnorm_abstract, len(gnorm_title))

                    for start, end, text, gene_id in genes:
                        if 0 <= start < end <= len(title.text):
                            part = title
                        else:
                            part = abstract
                            # we have to readjust the offset since GnormPlus provides
                            # offsets for title and abstract together
                            offset = len(title.text) + 1
                            start -= offset
                            end -= offset

                        for adjustment_offset, adjustment in adjustment_offsets:
                            if start > adjustment_offset:
                                start -= adjustment

                        # todo discussion which confidence value for gnormplus because there is no value supplied
                        ann = Entity(class_id=PRO_CLASS_ID, offset=start, text=text, confidence=0.5)
                        try:
                            norm_dict = {
                                ENTREZ_GENE_ID: gene_id,
                                UNIPROT_ID: genes_mapping[gene_id]
                            }
                        except KeyError:
                            norm_dict = {ENTREZ_GENE_ID: gene_id}

                        norm_string = ''  # todo normalized_text (stemming ... ?)
                        ann.normalisation_dict = norm_dict
                        ann.normalized_text = norm_string
                        if annotated:
                            part.annotations.append(ann)
                        else:
                            part.predicted_annotations.append(ann)
                else:
                    # todo this is not used for now anywhere, might need to be re-worked or excluded
                    # genes = gnorm.get_genes_for_text(part.text)
                    pass
