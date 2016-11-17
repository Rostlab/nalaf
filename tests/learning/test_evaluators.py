import unittest
from nalaf.structures.data import Dataset, Document, Part, Entity, Relation
from nalaf.learning.evaluators import Evaluator, MentionLevelEvaluator, DocumentLevelRelationEvaluator


STUB_E_ID_1 = 'e_x_1'
STUB_E_ID_2 = 'e_x_2'
STUB_R_ID_1 = 'r_x_1'


class TestMentionLevelEvaluator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # create a sample dataset1 (1) to test
        cls.dataset1 = Dataset()
        doc_1 = Document()

        text = '.... aaaa .... bbbb .... cccc .... dddd .... eeee .... ffff .... gggg .... hhhh .... jjjj'
        part_1 = Part(text)

        cls.dataset1.documents['doc_1'] = doc_1
        doc_1.parts['part_1'] = part_1

        exact_1 = Entity(STUB_E_ID_1, 5, 'aaaa')
        exact_1.subclass = 1
        exact_2 = Entity(STUB_E_ID_1, 55, 'ffff')
        exact_2.subclass = 2
        exact_3 = Entity(STUB_E_ID_1, 75, 'hhhh')
        exact_3.subclass = 2

        overlap_1_1 = Entity(STUB_E_ID_1, 25, 'cccc')
        overlap_1_1.subclass = 1
        overlap_1_2 = Entity(STUB_E_ID_1, 26, 'cc')
        overlap_1_2.subclass = 1

        overlap_2_1 = Entity(STUB_E_ID_1, 32, '.. ddd')
        overlap_2_1.subclass = 2
        overlap_2_2 = Entity(STUB_E_ID_1, 36, 'ddd ...')
        overlap_2_2.subclass = 2

        overlap_3_1 = Entity(STUB_E_ID_1, 65, 'gggg')
        overlap_3_1.subclass = 1
        overlap_3_2 = Entity(STUB_E_ID_1, 62, '.. gggg ..')
        overlap_3_2.subclass = 2

        missing_1 = Entity('e2', 45, 'eeee')
        missing_1.subclass = 1
        missing_2 = Entity('e2', 84, 'jjjj')
        missing_2.subclass = 1

        spurios = Entity('e2', 15, 'bbbb')
        spurios.subclass = 1

        part_1.annotations = [exact_1, exact_2, exact_3, overlap_1_1, overlap_2_1, overlap_3_1, missing_1, missing_2]
        part_1.predicted_annotations = [exact_1, exact_2, exact_3, overlap_1_2, overlap_2_2, overlap_3_2, spurios]


    def test_implements_evaluator_interface(self):
        self.assertIsInstance(MentionLevelEvaluator(), Evaluator)

    def test_exact_strictness(self):
        evaluator = MentionLevelEvaluator()
        evaluation = (evaluator.evaluate(self.dataset1))(MentionLevelEvaluator.TOTAL_LABEL)

        self.assertEqual(evaluation.tp, 3)  # the 3 exact matches
        self.assertEqual(evaluation.fp, 4)  # the 3 overlapping + 1 spurious
        self.assertEqual(evaluation.fn, 5)  # the 3 overlapping + 2 missing

        ret = evaluation.compute('exact')

        self.assertEqual(ret.precision, 3 / 7)
        self.assertEqual(ret.recall, 3 / 8)
        self.assertEqual(ret.f_measure, 2 * (3 / 7 * 3 / 8) / (3 / 7 + 3 / 8))

    def test_overlapping_strictness(self):
        evaluator = MentionLevelEvaluator()
        evaluation = (evaluator.evaluate(self.dataset1))(MentionLevelEvaluator.TOTAL_LABEL)

        self.assertEqual(evaluation.tp, 3)  # the 3 exact matches
        self.assertEqual(evaluation.fp - evaluation.fp_ov, 1)  # the 1 spurious
        self.assertEqual(evaluation.fn - evaluation.fn_ov, 2)  # the 2 missing
        self.assertEqual(evaluation.fp_ov, 3)  # the 3 overlapping
        self.assertEqual(evaluation.fn_ov, 3)  # the 3 overlapping

        ret = evaluation.compute('overlapping')

        self.assertEqual(ret.precision, 9 / 10)
        self.assertEqual(ret.recall, 9 / 11)
        self.assertAlmostEqual(ret.f_measure, 2 * (9 / 10 * 9 / 11) / (9 / 10 + 9 / 11), places=5)

    def test_half_overlapping_strictness(self):
        evaluator = MentionLevelEvaluator()
        evaluation = (evaluator.evaluate(self.dataset1))(MentionLevelEvaluator.TOTAL_LABEL)

        self.assertEqual(evaluation.tp, 3)  # the 3 exact matches
        self.assertEqual(evaluation.fp - evaluation.fp_ov, 1)  # the 1 spurious
        self.assertEqual(evaluation.fn - evaluation.fn_ov, 2)  # the 2 missing
        self.assertEqual(evaluation.fp_ov, 3)  # the 3 overlapping
        self.assertEqual(evaluation.fn_ov, 3)  # the 3 overlapping

        ret = evaluation.compute('half_overlapping')

        self.assertEqual(ret.precision, (3 + 6 / 2) / 10)
        self.assertEqual(ret.recall, (3 + 6 / 2) / 11)
        self.assertEqual(ret.f_measure, 2 * ((3 + 6 / 2) / 10 * (3 + 6 / 2) / 11) / ((3 + 6 / 2) / 10 + (3 + 6 / 2) / 11))

    def test_exception_on_equality_operator(self):
        ann_1 = Entity(STUB_E_ID_1, 1, 'text_1')
        ann_2 = Entity(STUB_E_ID_1, 2, 'text_2')

        Entity.equality_operator = 'not valid'
        self.assertRaises(ValueError, lambda: ann_1 == ann_2)

    def test_exception_on_strictness(self):
        evaluator = MentionLevelEvaluator()  # this is fine
        evaluation = (evaluator.evaluate(self.dataset1))(MentionLevelEvaluator.TOTAL_LABEL)  # this is fine

        self.assertRaises(ValueError, evaluation.compute, 'strictness not valid')

    def test_subclass_analysis(self):
        evaluator = MentionLevelEvaluator(subclass_analysis=True)
        evaluations = evaluator.evaluate(self.dataset1)

        self.assertEqual(evaluations(1).tp, 1)
        self.assertEqual(evaluations(2).tp, 2)

        self.assertEqual(evaluations(1).fp, 3)
        self.assertEqual(evaluations(2).fp, 1)

        self.assertEqual(evaluations(1).fn, 4)
        self.assertEqual(evaluations(2).fn, 1)

        self.assertEqual(evaluations(1).fp_ov, 2)
        self.assertEqual(evaluations(1).fn_ov, 2)
        self.assertEqual(evaluations(2).fp_ov, 1)
        self.assertEqual(evaluations(2).fn_ov, 1)

    # -------

    def test_DocumentLevelRelationEvaluator_default_entities_case_irrelevant(self):

        evaluator = DocumentLevelRelationEvaluator(rel_type=STUB_R_ID_1)

        dataset = Dataset()
        doc_1 = Document()
        part_1 = Part('_irrelevant_')
        dataset.documents['doc_1'] = doc_1
        doc_1.parts['part_1'] = part_1

        part_1.relations = [
            Relation(
                STUB_R_ID_1,
                Entity(STUB_E_ID_1, 0, "TOOL"),
                Entity(STUB_E_ID_2, 0, "maynard")
            ),
        ]

        # -

        part_1.predicted_relations = [
            # empty
        ]

        evals = evaluator.evaluate(dataset)
        evaluation = evals(STUB_R_ID_1)
        self.assertEqual(evaluation.tp, 0)
        computation = evals(STUB_R_ID_1).compute(strictness="exact")
        self.assertEqual(computation.f_measure, 0.0)

        # -

        part_1.predicted_relations = [
            Relation(
                STUB_R_ID_1,
                Entity(STUB_E_ID_1, 0, "TOOL"),
                Entity(STUB_E_ID_2, 0, "maynard")
            ),
        ]

        evals = evaluator.evaluate(dataset)
        evaluation = evals(STUB_R_ID_1)
        self.assertEqual(evaluation.tp, 1)
        computation = evals(STUB_R_ID_1).compute(strictness="exact")
        self.assertEqual(computation.f_measure, 1.0)

        # -

        part_1.predicted_relations = [
            Relation(
                STUB_R_ID_1,
                Entity(STUB_E_ID_1, 0, "tool"),
                Entity(STUB_E_ID_2, 0, "MAYNARD")
            ),
        ]

        evals = evaluator.evaluate(dataset)
        evaluation = evals(STUB_R_ID_1)
        self.assertEqual(evaluation.tp, 1)
        computation = evals(STUB_R_ID_1).compute(strictness="exact")
        self.assertEqual(computation.f_measure, 1.0)


    def test_DocumentLevelRelationEvaluator_order_irrelevant(self):

        evaluator = DocumentLevelRelationEvaluator(rel_type=STUB_R_ID_1)

        dataset = Dataset()
        doc_1 = Document()
        part_1 = Part('_irrelevant_')
        dataset.documents['doc_1'] = doc_1
        doc_1.parts['part_1'] = part_1

        part_1.relations = [
            Relation(
                STUB_R_ID_1,
                Entity(STUB_E_ID_1, 0, "TOOL"),
                Entity(STUB_E_ID_2, 0, "maynard")
            ),
        ]

        # -

        part_1.predicted_relations = [
            Relation(
                STUB_R_ID_1,
                Entity(STUB_E_ID_2, 0, "maynard"),
                Entity(STUB_E_ID_1, 0, "TOOL")
            ),
        ]

        evals = evaluator.evaluate(dataset)
        evaluation = evals(STUB_R_ID_1)
        self.assertEqual(evaluation.tp, 1)
        self.assertEqual(evaluation.fn, 0)
        self.assertEqual(evaluation.fp, 0)
        computation = evals(STUB_R_ID_1).compute(strictness="exact")
        self.assertEqual(computation.f_measure, 1.0)


    def test_DocumentLevelRelationEvaluator_false_positives(self):

        evaluator = DocumentLevelRelationEvaluator(rel_type=STUB_R_ID_1)

        dataset = Dataset()
        doc_1 = Document()
        part_1 = Part('_irrelevant_ PART *1*')
        dataset.documents['doc_1'] = doc_1
        doc_1.parts['part_1'] = part_1

        part_2 = Part('_irrelevant_ PART *2*')
        dataset.documents['doc_1'] = doc_1
        doc_1.parts['part_2'] = part_2

        part_1.relations = [
            Relation(STUB_R_ID_1, Entity(STUB_E_ID_1, 0, "TOOL"), Entity(STUB_E_ID_2, 0, "Maynard")),
        ]

        # -

        part_2.predicted_relations = [
            Relation(STUB_R_ID_1, Entity(STUB_E_ID_2, 0, "TOOL"), Entity(STUB_E_ID_1, 0, "Snoop Dog")),
        ]

        evals = evaluator.evaluate(dataset)
        evaluation = evals(STUB_R_ID_1)
        self.assertEqual(evaluation.tp, 0)
        self.assertEqual(evaluation.fn, 1)
        self.assertEqual(evaluation.fp, 1)
        computation = evals(STUB_R_ID_1).compute(strictness="exact")
        self.assertEqual(computation.f_measure, 0.0)


    def test_DocumentLevelRelationEvaluator_parts_irrelevant(self):

        evaluator = DocumentLevelRelationEvaluator(rel_type=STUB_R_ID_1)

        dataset = Dataset()
        doc_1 = Document()
        part_1 = Part('_irrelevant_ PART *1*')
        dataset.documents['doc_1'] = doc_1
        doc_1.parts['part_1'] = part_1

        part_2 = Part('_irrelevant_ PART *2*')
        dataset.documents['doc_1'] = doc_1
        doc_1.parts['part_2'] = part_2

        part_1.relations = [
            Relation(STUB_R_ID_1, Entity(STUB_E_ID_1, 0, "TOOL"), Entity(STUB_E_ID_2, 0, "maynard")),
        ]

        # -

        part_2.predicted_relations = [
            Relation(STUB_R_ID_1, Entity(STUB_E_ID_2, 0, "maynard"), Entity(STUB_E_ID_1, 0, "TOOL")),
        ]

        evals = evaluator.evaluate(dataset)
        evaluation = evals(STUB_R_ID_1)
        self.assertEqual(evaluation.tp, 1)
        self.assertEqual(evaluation.fn, 0)
        self.assertEqual(evaluation.fp, 0)
        computation = evals(STUB_R_ID_1).compute(strictness="exact")
        self.assertEqual(computation.f_measure, 1.0)


    def test_DocumentLevelRelationEvaluator_repeated_relations_irrelevant(self):

        evaluator = DocumentLevelRelationEvaluator(rel_type=STUB_R_ID_1)

        dataset = Dataset()
        doc_1 = Document()
        part_1 = Part('_irrelevant_')
        dataset.documents['doc_1'] = doc_1
        doc_1.parts['part_1'] = part_1

        part_1.relations = [
            Relation(STUB_R_ID_1, Entity(STUB_E_ID_1, 0, "TOOL"), Entity(STUB_E_ID_2, 0, "maynard")),

            Relation(STUB_R_ID_1, Entity(STUB_E_ID_1, 0, "TOOL"), Entity(STUB_E_ID_2, 0, "Danny Carey")),

            Relation(STUB_R_ID_1, Entity(STUB_E_ID_1, 1, "TOOL"), Entity(STUB_E_ID_2, 1, "Danny Carey")),
        ]

        # -

        part_1.predicted_relations = [
            Relation(STUB_R_ID_1, Entity(STUB_E_ID_1, 0, "TOOL"), Entity(STUB_E_ID_2, 0, "maynard")),
            Relation(STUB_R_ID_1, Entity(STUB_E_ID_1, 1, "TOOL"), Entity(STUB_E_ID_2, 1, "maynard")),
        ]

        evals = evaluator.evaluate(dataset)
        evaluation = evals(STUB_R_ID_1)
        self.assertEqual(evaluation.tp, 1)
        self.assertEqual(evaluation.fn, 1)
        self.assertEqual(evaluation.fp, 0)
        computation = evals(STUB_R_ID_1).compute(strictness="exact")
        self.assertEqual(computation.f_measure, 0.6666666666666666)

        # -

        part_1.predicted_relations = [
            Relation(STUB_R_ID_1, Entity(STUB_E_ID_1, 2, "TOOL"), Entity(STUB_E_ID_2, 2, "maynard")),
            Relation(STUB_R_ID_1, Entity(STUB_E_ID_1, 3, "TOOL"), Entity(STUB_E_ID_2, 3, "maynard")),

            Relation(STUB_R_ID_1, Entity(STUB_E_ID_1, 4, "TOOL"), Entity(STUB_E_ID_2, 4, "Danny Carey")),
        ]

        evals = evaluator.evaluate(dataset)
        evaluation = evals(STUB_R_ID_1)
        self.assertEqual(evaluation.tp, 2)
        self.assertEqual(evaluation.fn, 0)
        self.assertEqual(evaluation.fp, 0)
        computation = evals(STUB_R_ID_1).compute(strictness="exact")
        self.assertEqual(computation.f_measure, 1.0)


    def test_DocumentLevelRelationEvaluator_normalized_entities(self):

        evaluator = DocumentLevelRelationEvaluator(rel_type=STUB_R_ID_1, entity_map_fun=DocumentLevelRelationEvaluator.COMMON_ENTITY_MAP_FUNS['normalized_fun']('n_1'))

        dataset = Dataset()
        doc_1 = Document()
        part_1 = Part('_irrelevant_')
        dataset.documents['doc_1'] = doc_1
        doc_1.parts['part_1'] = part_1

        part_1.relations = [
            Relation(
                STUB_R_ID_1,
                Entity(STUB_E_ID_1, 0, "Tool", norm={"n_1": 1964}),
                Entity(STUB_E_ID_2, 0, "Maynard", norm={"n_1": 1961})),
        ]

        # -

        part_1.predicted_relations = [
            Relation(
                STUB_R_ID_1,
                Entity(STUB_E_ID_1, 0, "Tool"),
                Entity(STUB_E_ID_2, 0, "Maynard", norm={"n_x": 1961})),

            Relation(
                STUB_R_ID_1,
                Entity(STUB_E_ID_1, 0, "Tool", norm={"n_1": 666}),
                Entity(STUB_E_ID_2, 0, "Maynard", norm={"n_1": 1961})),

            Relation(
                STUB_R_ID_1,
                Entity(STUB_E_ID_1, 0, "Tool", norm={"n_another_key": 1964}),
                Entity(STUB_E_ID_2, 0, "Maynard", norm={"n_another_key": 1961})),
        ]

        evals = evaluator.evaluate(dataset)
        evaluation = evals(STUB_R_ID_1)
        self.assertEqual(evaluation.tp, 0)
        self.assertEqual(evaluation.fn, 1)
        self.assertEqual(evaluation.fp, 3)
        computation = evals(STUB_R_ID_1).compute(strictness="exact")
        self.assertEqual(computation.f_measure, 0.0)

        # -

        part_1.predicted_relations = [
            Relation(
                STUB_R_ID_1,
                Entity(STUB_E_ID_1, 0, "Tool band", norm={"n_1": 1964}),
                Entity(STUB_E_ID_2, 0, "Maynard James Keenan", norm={"n_1": 1961})),
        ]

        evals = evaluator.evaluate(dataset)
        evaluation = evals(STUB_R_ID_1)
        self.assertEqual(evaluation.tp, 1)
        self.assertEqual(evaluation.fn, 0)
        self.assertEqual(evaluation.fp, 0)
        computation = evals(STUB_R_ID_1).compute(strictness="exact")
        self.assertEqual(computation.f_measure, 1.0)


    def _create_basic_dataset(self):
        dataset = Dataset()
        doc_1 = Document()
        part_1 = Part('_irrelevant_')
        dataset.documents['doc_1'] = doc_1
        doc_1.parts['part_1'] = part_1
        return (dataset, part_1)


    def test_DocumentLevelRelationEvaluator_arbitrary_relation_equiv_fun_order_does_not_matter(self):

        entity_map_fun = (lambda e: "SAME")

        def relation_equiv_fun(gold, pred):
            print('gold:', gold, ' <---> ', 'pred:', pred)
            return gold == pred

        r1 = Relation(STUB_R_ID_1, Entity(STUB_E_ID_1, 0, "yin"), Entity(STUB_E_ID_2, 0, "yan"))
        r2 = Relation(STUB_R_ID_1, Entity(STUB_E_ID_1, 0, "yan"), Entity(STUB_E_ID_2, 0, "yin"))

        self.assertTrue(relation_equiv_fun(r1.map(entity_map_fun), r1.map(entity_map_fun)))
        self.assertTrue(relation_equiv_fun(r1.map(entity_map_fun), r2.map(entity_map_fun)))
        self.assertTrue(relation_equiv_fun(r2.map(entity_map_fun), r1.map(entity_map_fun)))

        evaluator = DocumentLevelRelationEvaluator(STUB_R_ID_1, entity_map_fun, relation_equiv_fun)

        (dataset, part) = self._create_basic_dataset()

        # -

        part.relations = [r1]
        part.predicted_relations = [r1]

        evals = evaluator.evaluate(dataset)
        evaluation = evals(STUB_R_ID_1)
        print(evaluation)
        self.assertEqual(evaluation.tp, 1)
        self.assertEqual(evaluation.fn, 0)
        self.assertEqual(evaluation.fp, 0)
        computation = evals(STUB_R_ID_1).compute(strictness="exact")
        self.assertEqual(computation.f_measure, 1.0)


    def test_DocumentLevelRelationEvaluator_arbitrary_relation_equiv_fun_order_matters(self):

        entity_map_fun = (lambda e: e.text)

        def relation_equiv_fun(gold, pred):
            print('gold:', gold, ' <---> ', 'pred:', pred)
            return gold < pred

        r1 = Relation(STUB_R_ID_1, Entity(STUB_E_ID_1, 0, "1"), Entity(STUB_E_ID_2, 0, "2"))
        r2 = Relation(STUB_R_ID_1, Entity(STUB_E_ID_1, 0, "2"), Entity(STUB_E_ID_2, 0, "1"))

        # r1 not equiv r1 because this IS NOT equals (r1 not < r1)
        self.assertFalse(relation_equiv_fun(r1.map(entity_map_fun), r1.map(entity_map_fun)))
        # r1 < r2
        self.assertTrue(relation_equiv_fun(r1.map(entity_map_fun), r2.map(entity_map_fun)))
        # r2 not < r1
        self.assertFalse(relation_equiv_fun(r2.map(entity_map_fun), r1.map(entity_map_fun)))

        evaluator = DocumentLevelRelationEvaluator(STUB_R_ID_1, entity_map_fun, relation_equiv_fun)

        (dataset, part) = self._create_basic_dataset()

        # -

        part.relations = [r1]
        part.predicted_relations = [r1]

        evals = evaluator.evaluate(dataset)
        evaluation = evals(STUB_R_ID_1)
        print(evaluation)
        self.assertEqual(evaluation.tp, 0)
        self.assertEqual(evaluation.fn, 1)
        self.assertEqual(evaluation.fp, 1)
        computation = evals(STUB_R_ID_1).compute(strictness="exact")
        self.assertEqual(computation.f_measure, 0.0)

        # -

        part.relations = [r1]
        part.predicted_relations = [r2]

        evals = evaluator.evaluate(dataset)
        evaluation = evals(STUB_R_ID_1)
        print(evaluation)
        self.assertEqual(evaluation.tp, 1)
        self.assertEqual(evaluation.fn, 0)
        self.assertEqual(evaluation.fp, 0)
        computation = evals(STUB_R_ID_1).compute(strictness="exact")
        self.assertEqual(computation.f_measure, 1.0)

        # -

        part.relations = [r2]
        part.predicted_relations = [r1]

        evals = evaluator.evaluate(dataset)
        evaluation = evals(STUB_R_ID_1)
        self.assertEqual(evaluation.tp, 0)
        self.assertEqual(evaluation.fn, 1)
        self.assertEqual(evaluation.fp, 1)
        computation = evals(STUB_R_ID_1).compute(strictness="exact")
        self.assertEqual(computation.f_measure, 0.0)


if __name__ == '__main__':
    unittest.main()
