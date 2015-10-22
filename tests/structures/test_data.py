import unittest
from nala.structures.data import Dataset, Document, Part, Token, Label, Entity
from nala.utils import MUT_CLASS_ID
from nala.preprocessing.spliters import NLTKSplitter
# from preprocessing.tokenizers import TmVarTokenizer


class TestDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = Dataset()
        cls.doc = Document()
        cls.dataset.documents['testid'] = cls.doc

        # TEXT = "123 45678"
        # POS  = "012345678"
        # ANN1 = " X       "
        # ANN2 = "     XXX "
        # PAR1 = "XXX      "
        # PAR1 = "    XXXXX"

        part1 = Part('123')
        part2 = Part('45678')
        ann1 = Entity(class_id='e_2', offset=1, text='2', confidence=0)
        ann2 = Entity(class_id='e_2', offset=1, text='567', confidence=1)
        ann1.subclass = 0
        ann2.subclass = 2
        part1.annotations.append(ann1)
        part2.annotations.append(ann2)
        cls.doc.parts['s1h1'] = part1
        cls.doc.parts['s2p1'] = part2

        doc2 = Document()
        doc3 = Document().parts['someid'] = Part('marmor stein und eisen')
        cls.dataset2 = Dataset()
        cls.dataset2.documents['newid'] = doc3
        cls.dataset2.documents['testid'] = doc2

    def test_repr_full(self):
        print(str(self.dataset))

    def test_overlaps_with_mention(self):
        # True states
        self.assertTrue(self.doc.overlaps_with_mention(5, 5))
        self.assertTrue(self.doc.overlaps_with_mention(6, 8))
        self.assertTrue(self.doc.overlaps_with_mention(4, 5))
        self.assertTrue(self.doc.overlaps_with_mention(1, 7))

        # False states
        self.assertFalse(self.doc.overlaps_with_mention(3, 4))
        self.assertFalse(self.doc.overlaps_with_mention(0, 0))
        self.assertFalse(self.doc.overlaps_with_mention(2, 4))

    def test_get_title(self):
        self.assertEquals(self.doc.get_title(), '123')

    def test_get_text(self):
        self.assertEquals(self.doc.get_text(), '123 45678')

    def test_get_body(self):
        self.assertEquals(self.doc.get_body(), '45678')

    def test_get_size(self):
        self.assertEquals(self.doc.get_size(), 9)

    def test_extend_dataset(self):
        print(self.dataset)
        print("\n\n\n")
        self.dataset.extend_dataset(self.dataset2)
        print(self.dataset)

    def test_delete_subclass_annotations(self):
        self.dataset.delete_subclass_annotations(0)
        self.assertEqual(len(list(self.dataset.annotations())), 1)


class TestDocument(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        text1 = "Flowers in the Rain. Are absolutely marvellous. Though i would say this text is stupid. Cheers!"

        part1 = Part(text1)
        doc = Document()
        doc.parts['firstpart'] = part1
        dataset = Dataset()
        dataset.documents['firstdocument'] = doc

        NLTKSplitter().split(dataset)
        # TmVarTokenizer().tokenize(dataset)
        cls.data = dataset
        cls.testpart = dataset.documents['firstdocument'].parts['firstpart']

    def test_get_size(self):
        self.assertEqual(self.testpart.get_size(), 95)

    def test_get_sentence_string_array(self):
        self.assertEqual(self.testpart.get_sentence_string_array(),
                         ["Flowers in the Rain.", "Are absolutely marvellous.",
                          "Though i would say this text is stupid.", "Cheers!"])


class TestToken(unittest.TestCase):
    def test_init(self):
        pass  # TODO

    def test_repr(self):
        pass  # TODO


class TestFeatureDictionary(unittest.TestCase):
    def test_setitem(self):
        pass  # TODO


class TestEntity(unittest.TestCase):
    def test_init(self):
        pass  # TODO

    def test_repr(self):
        pass  # TODO

    def test_eq(self):
        pass  # TODO


class TestLabel(unittest.TestCase):
    def test_repr(self):
        pass  # TODO

    def test_init(self):
        pass  # TODO


class TestPart(unittest.TestCase):
    def test_init(self):
        pass  # TODO

    def test_iter(self):
        pass  # TODO


class TestMentionLevel(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        # create a sample dataset to test
        cls.dataset = Dataset()
        part = Part('some text c.A100G p.V100Q some text')
        part.sentences = [[Token('some', 0), Token('text', 5), Token('c', 10), Token('.', 11), Token('A', 12),
                           Token('100', 13), Token('G', 16), Token('p', 18), Token('.', 19), Token('V', 20),
                           Token('100', 21), Token('Q', 24), Token('some', 26), Token('text', 31)]]

        predicted_labels = ['O', 'O', 'B', 'I', 'I', 'I', 'E', 'A', 'I', 'I', 'I', 'E', 'O', 'O']

        for index, label in enumerate(predicted_labels):
            part.sentences[0][index].predicted_labels = [Label(label)]

        cls.dataset.documents['doc_1'] = Document()
        cls.dataset.documents['doc_1'].parts['p1'] = part

        part = Part('test edge case DNA A927B test')
        part.sentences = [[Token('test', 0), Token('edge', 5), Token('case', 10), Token('DNA', 15),
                           Token('A', 19), Token('927', 20), Token('B', 23), Token('test', 25)]]

        predicted_labels = ['O', 'O', 'O', 'O', 'M', 'P', 'M', 'O']

        for index, label in enumerate(predicted_labels):
            part.sentences[0][index].predicted_labels = [Label(label)]

        cls.dataset.documents['doc_1'].parts['p2'] = part

    def test_form_predicted_annotations(self):
        self.dataset.form_predicted_annotations(MUT_CLASS_ID)

        part = self.dataset.documents['doc_1'].parts['p1']

        self.assertEqual(len(part.predicted_annotations), 2)

        self.assertEqual(part.predicted_annotations[0].text, 'c.A100G')
        self.assertEqual(part.predicted_annotations[0].offset, 10)

        self.assertEqual(part.predicted_annotations[1].text, 'p.V100Q')
        self.assertEqual(part.predicted_annotations[1].offset, 18)

        part = self.dataset.documents['doc_1'].parts['p2']

        self.assertEqual(len(part.predicted_annotations), 1)

        self.assertEqual(part.predicted_annotations[0].text, 'A927B')
        self.assertEqual(part.predicted_annotations[0].offset, 19)


if __name__ == '__main__':
    unittest.main()
