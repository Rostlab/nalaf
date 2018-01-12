import unittest
from nalaf.structures.data import Dataset, Document, Part, Token, Label, Entity
from nalaf.preprocessing.spliters import NLTKSplitter
from nalaf import print_verbose, print_debug
from nalaf.preprocessing.edges import SentenceDistanceEdgeGenerator
from nalaf.preprocessing.spliters import NLTKSplitter
from nalaf.preprocessing.tokenizers import NLTK_TOKENIZER
import sys


STUB_ENTITY_CLASS_ID_1 = 'e_x_1'
STUB_ENTITY_CLASS_ID_2 = 'e_x_2'
STUB_RELATION_CLASS_ID_2 = 'r_x'


class TestSentenceDistanceEdgeGenerator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dataset = Dataset()
        cls.doc = Document()
        cls.dataset.documents['testid'] = cls.doc

        part1 = Part('Sentence 1: e_1_yolo may be related to e_2_tool plus hey, e_2_coco. Sentence 2: e_1_nin. Sentence 3: e_2_musk. Sentence 4: nothing')

        entities = [
            # Sent 1
            Entity(class_id=STUB_ENTITY_CLASS_ID_1, offset=12, text='e_1_yolo', confidence=0),
            Entity(class_id=STUB_ENTITY_CLASS_ID_2, offset=39, text='e_2_tool', confidence=0),
            Entity(class_id=STUB_ENTITY_CLASS_ID_2, offset=58, text='e_2_coco', confidence=0),
            # Sent 2
            Entity(class_id=STUB_ENTITY_CLASS_ID_1, offset=80, text='e_1_nin', confidence=0),
            # Sent 3
            Entity(class_id=STUB_ENTITY_CLASS_ID_2, offset=101, text='e_2_musk', confidence=0),
            # Sent 4

        ]

        for e in entities:
            part1.annotations.append(e)

        cls.doc.parts['s1h1'] = part1

        cls.splitter = NLTKSplitter()
        cls.tokenizer = NLTK_TOKENIZER

        cls.splitter.split(cls.dataset)
        cls.tokenizer.tokenize(cls.dataset)

        # assert False, str(list(cls.dataset.sentences()))
        assert 4 == len(list(cls.dataset.sentences())), str(list(cls.dataset.sentences()))


    def test_distance_0(self):
        edge_generator = SentenceDistanceEdgeGenerator(STUB_ENTITY_CLASS_ID_1, STUB_ENTITY_CLASS_ID_2, STUB_RELATION_CLASS_ID_2, distance=0)
        edge_generator.generate(self.dataset)
        num_edges = len(list(self.dataset.edges()))

        self.assertEqual(num_edges, 1 + 1, "\n"+"\n".join(str(e) for e in self.dataset.edges()))


    def test_distance_1(self):
        edge_generator = SentenceDistanceEdgeGenerator(STUB_ENTITY_CLASS_ID_1, STUB_ENTITY_CLASS_ID_2, STUB_RELATION_CLASS_ID_2, distance=1)
        edge_generator.generate(self.dataset)
        num_edges = len(list(self.dataset.edges()))

        self.assertEqual(num_edges, 1 + 1 + 1, "\n"+"\n".join(str(e) for e in self.dataset.edges()))


    def test_distance_2(self):
        edge_generator = SentenceDistanceEdgeGenerator(STUB_ENTITY_CLASS_ID_1, STUB_ENTITY_CLASS_ID_2, STUB_RELATION_CLASS_ID_2, distance=2)
        edge_generator.generate(self.dataset)
        num_edges = len(list(self.dataset.edges()))

        self.assertEqual(num_edges, 1, "\n"+"\n".join(str(e) for e in self.dataset.edges()))


    def test_distance_infinite(self):
        edge_generator = SentenceDistanceEdgeGenerator(STUB_ENTITY_CLASS_ID_1, STUB_ENTITY_CLASS_ID_2, STUB_RELATION_CLASS_ID_2, distance=sys.maxsize)
        edge_generator.generate(self.dataset)
        num_edges = len(list(self.dataset.edges()))

        self.assertEqual(num_edges, 0, "\n"+"\n".join(str(e) for e in self.dataset.edges()))


if __name__ == '__main__':
    unittest.main()
