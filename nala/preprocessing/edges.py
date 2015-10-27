import abc
from nala.structures.data import Edge

class EdgeGenerator:
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
        :type dataset: nala.structures.data.Dataset
        """
        return


class SimpleEdgeGenerator(EdgeGenerator):
    """
    Simple implementation of generating edges between the two entities
    if they are contained in the same sentence.

    Implements the abstract class EdgeGenerator.

    :type entity1_class: str
    :type entity2_class: str
    :type relation_type: str
    """

    def __init__(self, entity1_class, entity2_class, relation_type):
        self.entity1_class = entity1_class
        self.entity2_class = entity2_class
        self.relation_type = relation_type

    def generate(self, dataset):
        from itertools import product
        for part in dataset.parts():
            for ann_1, ann_2 in product(
                    (ann for ann in part.annotations if ann.class_id == self.entity1_class),
                    (ann for ann in part.annotations if ann.class_id == self.entity2_class)):
                index_1 = part.get_sentence_index_for_annotation(ann_1)
                index_2 = part.get_sentence_index_for_annotation(ann_2)
                if index_1 == index_2 and index_1 != None:
                    part.edges.append(
                        Edge(ann_1, ann_2, self.relation_type,
                        part.get_sentence_string_array()[index_1]))
