import abc
from nalaf.structures.data import Edge
from itertools import product, chain


class EdgeGenerator(object):
    """
    Abstract class for generating edges between two entities. Each edge represents
    a possible relationship between the two entities
    Subclasses that inherit this class should:
    * Be named [Name]EdgeGenerator
    * Implement the abstract method generate
    * Append new items to the list field "edges" of each Part in the dataset
    """

    @abc.abstractmethod
    def generate(self, dataset):
        """
        :type dataset: nalaf.structures.data.Dataset
        """
        return


class SentenceDistanceEdgeGenerator(EdgeGenerator):
    """
    Simple implementation of generating edges between the two entities
    if they are #`distance` sentences away (always same part)
    """

    def __init__(self, entity1_class, entity2_class, relation_type, distance, use_predicted_entities=True):
        self.entity1_class = entity1_class
        self.entity2_class = entity2_class
        self.relation_type = relation_type
        self.distance = distance
        self.use_predicted_entities = use_predicted_entities
        self.part_entities = (lambda part: chain(part.annotations, part.predicted_annotations) if self.use_predicted_entities else part.annotations)

    def generate(self, dataset):


        for part in dataset.parts():
            part.edges = []

            for ann_1, ann_2 in product(
                    (ann for ann in self.part_entities(part) if ann.class_id == self.entity1_class),
                    (ann for ann in self.part_entities(part) if ann.class_id == self.entity2_class)):

                sent_index_1 = part.get_sentence_index_for_annotation(ann_1)
                sent_index_2 = part.get_sentence_index_for_annotation(ann_2)

                pair_distance = abs(sent_index_1 - sent_index_2)

                if pair_distance == self.distance:
                    part.edges.append(
                        Edge(ann_1, ann_2, self.relation_type, sent_index_1, part))


class SimpleEdgeGenerator(SentenceDistanceEdgeGenerator):
    """
    Simple implementation of generating edges between the two entities
    if they are contained in the same sentence.

    **It uses both the _gold_ annotations and _predicted_ annotations.**

    """

    def __init__(self, entity1_class, entity2_class, relation_type):
        import warnings
        warnings.warn('Use `SentenceDistanceEdgeGenerator` directly. This will be deleted', DeprecationWarning)

        super().__init__(entity1_class, entity2_class, relation_type, distance=0, use_predicted_entities=True)


    def generate(self, dataset):
        import warnings
        warnings.warn('Use `SentenceDistanceEdgeGenerator` directly. This will be deleted', DeprecationWarning)

        super().generate(dataset)


class WordFilterEdgeGenerator(EdgeGenerator):
    """
    Simple implementation of generating edges between the two entities
    if they are contained in the same sentence AND the sentence
    contains one of the trigger-like given words

    **It only uses the _gold_ annotations**

    """

    def __init__(self, entity1_class, entity2_class, relation_type, words):
        self.entity1_class = entity1_class
        self.entity2_class = entity2_class
        self.relation_type = relation_type
        self.words = words


    def generate(self, dataset):
        from itertools import product

        for part in dataset.parts():
            part.edges = []

            for ann_1, ann_2 in product(
                    (ann for ann in part.annotations if ann.class_id == self.entity1_class),
                    (ann for ann in part.annotations if ann.class_id == self.entity2_class)):

                index_1 = part.get_sentence_index_for_annotation(ann_1)
                index_2 = part.get_sentence_index_for_annotation(ann_2)

                if index_1 == index_2 and index_1 is not None:

                    for token in part.sentences[index_1]:

                        if token.word in self.words:
                            part.edges.append(
                                Edge(ann_1, ann_2, self.relation_type, index_1, part))
                            break
