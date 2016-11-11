import abc
from nalaf.structures.data import Edge


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


class SimpleEdgeGenerator(EdgeGenerator):
    """
    Simple implementation of generating edges between the two entities
    if they are contained in the same sentence.

    **It uses both the _gold_ annotations and _predicted_ annotations.**

    **It _does_ reset the edges: first they are emptied, then more are added**

    """

    def __init__(self, entity1_class, entity2_class, relation_type):
        self.entity1_class = entity1_class
        self.entity2_class = entity2_class
        self.relation_type = relation_type

    def generate(self, dataset):
        from itertools import product, chain

        for part in dataset.parts():
            part.edges = []  # TODO should we rewrite the edges?

            for ann_1, ann_2 in product(
                    (ann for ann in chain(part.annotations, part.predicted_annotations) if ann.class_id == self.entity1_class),
                    (ann for ann in chain(part.annotations, part.predicted_annotations) if ann.class_id == self.entity2_class)):

                index_1 = part.get_sentence_index_for_annotation(ann_1)
                index_2 = part.get_sentence_index_for_annotation(ann_2)

                if index_1 == index_2 and index_1 is not None:
                    part.edges.append(
                        Edge(ann_1, ann_2, self.relation_type, index_1, part))


class WordFilterEdgeGenerator(EdgeGenerator):
    """
    Simple implementation of generating edges between the two entities
    if they are contained in the same sentence AND the sentence
    contains one of the trigger-like given words

    **It only uses the _gold_ annotations**

    **It does _not_ reset the edges: it only adds more**

    """

    def __init__(self, entity1_class, entity2_class, relation_type, words):
        self.entity1_class = entity1_class
        self.entity2_class = entity2_class
        self.relation_type = relation_type
        self.words = words


    def generate(self, dataset):
        from itertools import product

        for part in dataset.parts():

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


class SimpleD1EdgeGenerator(EdgeGenerator):
    """
        TODO document me
    """
    # TODO this is a STUB, for now

    def __init__(self, entity1_class, entity2_class, relation_type):
        self.entity1_class = entity1_class
        self.entity2_class = entity2_class
        self.relation_type = relation_type

    def generate(self, dataset):

        for part in dataset.parts():
            pass
            # part.edges = []  # TODO leave the edges intact for now
