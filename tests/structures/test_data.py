import unittest
from nala.structures.data import Dataset, Document, Part, Token, Label, Annotation
from nala.utils import MUT_CLASS_ID


class TestDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = Dataset()
        cls.doc = Document()
        cls.dataset.documents['testid'] = cls.doc
        part1 = Part('123')
        part2 = Part('45678')
        ann1 = Annotation(class_id='e_2', offset=1, text='2', confidence=0)
        ann2 = Annotation(class_id='e_2', offset=4, text='567', confidence=1)
        ann1.subclass = 0
        ann2.subclass = 2
        part1.annotations.append(ann1)
        part2.annotations.append(ann2)
        cls.doc.parts['s1h1'] = part1
        cls.doc.parts['s2p1'] = part2
    
    def test_overlaps_with_mention(self):
        print(self.doc.get_text())
        # part1
        self.assertTrue(self.doc.overlaps_with_mention(4, 4))
        # self.assertTrue(self.doc.overlaps_with_mention(6))
        # self.assertFalse(self.doc.overlaps_with_mention(5))

        # part2 with offsets
        # 16 is len('Start test text.')
        # self.assertTrue(self.doc.overlaps_with_mention(5 + 16))
        # self.assertTrue(self.doc.overlaps_with_mention(6 + 16))
        # self.assertFalse(self.doc.overlaps_with_mention(4 + 16))
        # self.assertFalse(self.doc.overlaps_with_mention(7 + 16))
        # self.assertFalse(self.doc.overlaps_with_mention2(3, 3))
        # self.assertTrue(self.doc.overlaps_with_mention2(3, 4))

    def test_get_title(self):
        self.assertEquals(self.doc.get_title(), '123')

    def test_get_text(self):
        self.assertEquals(self.doc.get_text(), '123 45678')

    def test_get_body(self):
        self.assertEquals(self.doc.get_body(), '45678')

    def test_get_size(self):
        self.assertEquals(self.doc.get_size(), 9)


class TestDocument(unittest.TestCase):
    def test_key_value_parts(self):
        pass  # TODO

    def test_get_unique_mentions(self):
        pass  # TODO

    def test_get_size(self):
        pass  # TODO


class TestToken(unittest.TestCase):
    def test_init(self):
        pass  # TODO

    def test_repr(self):
        pass  # TODO


class TestFeatureDictionary(unittest.TestCase):
    def test_setitem(self):
        pass  # TODO


class TestAnnotation(unittest.TestCase):
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
