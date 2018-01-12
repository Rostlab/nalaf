import unittest
from nalaf.learning.taggers import StubRelationExtractor, StubSameSentenceRelationExtractor, StubSamePartRelationExtractor
from nalaf.preprocessing.edges import SentenceDistanceEdgeGenerator, CombinatorEdgeGenerator
from nalaf.structures.data import *
from nalaf.learning.evaluators import DocumentLevelRelationEvaluator


STUB_E_ID_1 = 'e_x_1'
STUB_E_ID_2 = 'e_x_2'
STUB_R_ID_1 = 'r_x_1'


class TestTaggers(unittest.TestCase):

    @classmethod
    def get_test_dataset(cls):
        dataset = Dataset()
        doc = Document()
        dataset.documents['testid'] = doc

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

        doc.parts['s1h1'] = part1

        return dataset


    def test_StubSameSentenceRelationExtractor(self):

        dataset = TestTaggers.get_test_dataset()

        annotator = StubSameSentenceRelationExtractor(STUB_E_ID_1, STUB_E_ID_2, relation_type=STUB_R_ID_1)
        annotator.annotate(dataset)
        # Assert that indeed 4 sentences were considered
        assert 4 == len(list(dataset.sentences())), str(list(dataset.sentences()))

        print("actu_rels", list(dataset.relations()))
        print("edges", list(dataset.edges()))
        print("pred_rels", list(dataset.predicted_relations()))

        evaluator = DocumentLevelRelationEvaluator(rel_type=STUB_R_ID_1)

        evals = evaluator.evaluate(dataset)
        evaluation = evals(STUB_R_ID_1)
        self.assertEqual(evaluation.tp, 1)
        self.assertEqual(evaluation.fn, 2)
        self.assertEqual(evaluation.fp, 1)
        computation = evals(STUB_R_ID_1).compute(strictness="exact")
        self.assertEqual(computation.f_measure, 0.4)


    def test_Stub_D0_plus_D1_RelationExtractor(self):

        dataset = TestTaggers.get_test_dataset()

        edge_generator_1 = SentenceDistanceEdgeGenerator(STUB_E_ID_1, STUB_E_ID_2, STUB_R_ID_1, distance=0, rewrite_edges=False)
        edge_generator_2 = SentenceDistanceEdgeGenerator(STUB_E_ID_1, STUB_E_ID_2, STUB_R_ID_1, distance=1, rewrite_edges=False)
        edge_generator = CombinatorEdgeGenerator(edge_generator_1, edge_generator_2)
        annotator = StubRelationExtractor(edge_generator)

        annotator.annotate(dataset)
        # Assert that indeed 4 sentences were considered
        assert 4 == len(list(dataset.sentences())), str(list(dataset.sentences()))

        # print("actu_rels", list(dataset.relations()))
        # print("edges", list(dataset.edges()))
        # print("pred_rels", list(dataset.predicted_relations()))

        evaluator = DocumentLevelRelationEvaluator(rel_type=STUB_R_ID_1)

        evals = evaluator.evaluate(dataset)
        evaluation = evals(STUB_R_ID_1)
        self.assertEqual(evaluation.tp, 3)
        self.assertEqual(evaluation.fn, 0)
        self.assertEqual(evaluation.fp, 2)
        computation = evals(STUB_R_ID_1).compute(strictness="exact")
        self.assertEqual(computation.f_measure, 0.7499999999999999)


    def test_StubSamePartRelationExtractor(self):

        dataset = TestTaggers.get_test_dataset()

        annotator = StubSamePartRelationExtractor(STUB_E_ID_1, STUB_E_ID_2, relation_type=STUB_R_ID_1)
        annotator.annotate(dataset)
        # Assert that indeed 4 sentences were considered
        assert 4 == len(list(dataset.sentences())), str(list(dataset.sentences()))

        print("actu_rels", list(dataset.relations()))
        print("edges", list(dataset.edges()))
        print("pred_rels", list(dataset.predicted_relations()))

        evaluator = DocumentLevelRelationEvaluator(rel_type=STUB_R_ID_1)

        evals = evaluator.evaluate(dataset)
        evaluation = evals(STUB_R_ID_1)
        self.assertEqual(evaluation.tp, 3)
        self.assertEqual(evaluation.fn, 0)
        self.assertEqual(evaluation.fp, 3)
        computation = evals(STUB_R_ID_1).compute(strictness="exact")
        self.assertEqual(computation.f_measure, 0.6666666666666666)


if __name__ == '__main__':
    unittest.main()
