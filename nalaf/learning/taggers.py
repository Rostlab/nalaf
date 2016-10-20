import abc
from nalaf.structures.data import Relation
import warnings


class Annotator:
    """
    Abstract class for Entity tagging or Relationship tagging.
    This forms a hierarchy, where Tagger and RelationExtractor are abstract
    subclasses of Annotator
    """

    def __init__(self, predicts_classes):
        self.predicts_classes = predicts_classes
        """a list of class IDs that this tagger can predict"""

    @abc.abstractmethod
    def annotate(self, dataset):
        """
        In general, do and add predictions to the dataset, i.e., annotate the dataset

        :type dataset: nalaf.structures.data.Dataset
        """


class Tagger(Annotator):
    """
    Abstract class for (entity-) tagging a dataset with predicted annotations.

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

    def tag(self, dataset):
        """
        :type dataset: nalaf.structures.data.Dataset
        """
        warnings.warn('Use rather the method: annotate', DeprecationWarning)
        self.annotate(dataset)

    @abc.abstractmethod
    def annotate(self, dataset):
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
        super().__init__(relation_type)

        self.entity1_class = entity1_class
        """class id of the first entity"""
        self.entity2_class = entity2_class
        """class id of the second entity"""
        self.relation_type = relation_type
        """the type of relation between the two entiies in predicts_classes"""

    def tag(self, dataset):
        """
        :type dataset: nalaf.structures.data.Dataset
        """
        warnings.warn('Use rather the method: annotate', DeprecationWarning)
        self.annotate(dataset)

    @abc.abstractmethod
    def annotate(self, dataset):
        """
        :type dataset: nalaf.structures.data.Dataset
        """


class StubSameSentenceRelationExtractor(RelationExtractor):

    def __init__(self, entity1_class, entity2_class, relation_type):
        super().__init__(entity1_class, entity2_class, relation_type)

    def tag(self, dataset):
        from itertools import product
        for part in dataset.parts():
            for ann_1, ann_2 in product(
                    (ann for ann in part.annotations if ann.class_id == self.entity1_class),
                    (ann for ann in part.annotations if ann.class_id == self.entity2_class)):
                if part.get_sentence_index_for_annotation(ann_1) == part.get_sentence_index_for_annotation(ann_2):
                    part.predicted_relations.append(
                        Relation(ann_1.offset, ann_2.offset, ann_1.text, ann_2.text, self.relation_type))
