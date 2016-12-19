import unittest
from nalaf.learning.taggers import StubSameSentenceRelationExtractor, StubSamePartRelationExtractor
from nalaf.structures.data import *
from nalaf.learning.evaluators import DocumentLevelRelationEvaluator


STUB_E_ID_1 = 'e_x_1'
STUB_E_ID_2 = 'e_x_2'
STUB_R_ID_1 = 'r_x_1'


class TestTaggers(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dataset = Dataset()
        cls.doc = Document()
        cls.dataset.documents['testid'] = cls.doc

        part1 = Part('Sentence 1: e_1_yolo may be related to e_2_tool plus hey, e_2_coco. Sentence 2: e_1_nin. Sentence 3: e_2_musk. Sentence 4: nothing')

        entities = [
            # Sent 1
            Entity(class_id=STUB_E_ID_1, offset=12, text='e_1_yolo', confidence=0),
            Entity(class_id=STUB_E_ID_2, offset=39, text='e_2_tool', confidence=0),
            Entity(class_id=STUB_E_ID_2, offset=58, text='e_2_coco', confidence=0),
            # Sent 2
            Entity(class_id=STUB_E_ID_1, offset=80, text='e_1_nin', confidence=0),
            # Sent 3
            Entity(class_id=STUB_E_ID_2, offset=101, text='e_2_musk', confidence=0),
            # Sent 4

        ]

        for e in entities:
            part1.annotations.append(e)

        relations = [
            # Same sentence -- Internal edge generator will create 2 edges out of Sent 1 but only 1 Relation is real
            Relation(STUB_R_ID_1, entities[0], entities[1]),
            # Between different sentences
            Relation(STUB_R_ID_1, entities[1], entities[3]),
            Relation(STUB_R_ID_1, entities[3], entities[4]),
        ]

        for r in relations:
            part1.relations.append(r)

        cls.doc.parts['s1h1'] = part1


    def test_StubSameSentenceRelationExtractor(self):

        annotator = StubSameSentenceRelationExtractor(STUB_E_ID_1, STUB_E_ID_2, relation_type=STUB_R_ID_1)
        annotator.annotate(self.dataset)
        # Assert that indeed 4 sentences were considered
        assert 4 == len(list(self.dataset.sentences())), str(list(self.dataset.sentences()))

        print("actu_rels", list(self.dataset.relations()))
        print("edges", list(self.dataset.edges()))
        print("pred_rels", list(self.dataset.predicted_relations()))

        evaluator = DocumentLevelRelationEvaluator(rel_type=STUB_R_ID_1)

        evals = evaluator.evaluate(self.dataset)
        evaluation = evals(STUB_R_ID_1)
        self.assertEqual(evaluation.tp, 1)
        self.assertEqual(evaluation.fn, 2)
        self.assertEqual(evaluation.fp, 1)
        computation = evals(STUB_R_ID_1).compute(strictness="exact")
        self.assertEqual(computation.f_measure, 0.4)


    def test_StubSamePartRelationExtractor(self):

        annotator = StubSamePartRelationExtractor(STUB_E_ID_1, STUB_E_ID_2, relation_type=STUB_R_ID_1)
        annotator.annotate(self.dataset)
        # Assert that indeed 4 sentences were considered
        assert 4 == len(list(self.dataset.sentences())), str(list(self.dataset.sentences()))

        print("actu_rels", list(self.dataset.relations()))
        print("edges", list(self.dataset.edges()))
        print("pred_rels", list(self.dataset.predicted_relations()))

        evaluator = DocumentLevelRelationEvaluator(rel_type=STUB_R_ID_1)

        evals = evaluator.evaluate(self.dataset)
        evaluation = evals(STUB_R_ID_1)
        self.assertEqual(evaluation.tp, 3)
        self.assertEqual(evaluation.fn, 0)
        self.assertEqual(evaluation.fp, 3)
        computation = evals(STUB_R_ID_1).compute(strictness="exact")
        self.assertEqual(computation.f_measure, 0.6666666666666666)


if __name__ == '__main__':
    unittest.main()
