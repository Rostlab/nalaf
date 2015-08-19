import unittest
from nala.structures.data import Dataset, Document, Part, Token, Label
from nala.utils import MUT_CLASS_ID


class TestDataset(unittest.TestCase):
    def test_parts(self):
        self.fail()  # TODO

    def test_annotations(self):
        self.fail()  # TODO

    def test_predicted_annotations(self):
        self.fail()  # TODO

    def test_sentences(self):
        self.fail()  # TODO

    def test_tokens(self):
        self.fail()  # TODO

    def test_partids_with_parts(self):
        self.fail()  # TODO

    def test_annotations_with_partids(self):
        self.fail()  # TODO

    def test_all_annotations_with_ids(self):
        self.fail()  # TODO

    def test_clean_nl_definitions(self):
        self.fail()  # TODO

    def test_stats(self):
        self.fail()  # TODO


class TestDocument(unittest.TestCase):
    def test_key_value_parts(self):
        self.fail()  # TODO

    def test_get_unique_mentions(self):
        self.fail()  # TODO

    def test_get_size(self):
        self.fail()  # TODO


class TestToken(unittest.TestCase):
    def test_init(self):
        self.fail()  # TODO

    def test_repr(self):
        self.fail()  # TODO


class TestFeatureDictionary(unittest.TestCase):
    def test_setitem(self):
        self.fail()  # TODO


class TestAnnotation(unittest.TestCase):
    def test_init(self):
        self.fail()  # TODO

    def test_repr(self):
        self.fail()  # TODO

    def test_eq(self):
        self.fail()  # TODO


class TestLabel(unittest.TestCase):
    def test_repr(self):
        self.fail()  # TODO

    def test_init(self):
        self.fail()  # TODO


class TestPart(unittest.TestCase):
    def test_init(self):
        self.fail()  # TODO

    def test_iter(self):
        self.fail()  # TODO


class TestMentionLevel(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        # create a sample dataset to test
        cls.dataset = Dataset()
        part = Part('some text c.A100G p.V100Q some text')
        part.sentences = [[Token('some'), Token('text'), Token('c'), Token('.'), Token('A'), Token('100'), Token('G'),
                           Token('p'), Token('.'), Token('V'), Token('100'), Token('Q'), Token('some'), Token('text')]]

        predicted_labels = ['O', 'O', 'B', 'I', 'I', 'I', 'E', 'B', 'I', 'I', 'I', 'E', 'O', 'O']

        for index, label in enumerate(predicted_labels):
            part.sentences[0][index].predicted_labels = [Label(label)]

        cls.dataset.documents['doc_1'] = Document()
        cls.dataset.documents['doc_1'].parts['p1'] = part

    def test_form_predicted_annotations(self):
        self.dataset.form_predicted_annotations(MUT_CLASS_ID)

        part = self.dataset.documents['doc_1'].parts['p1']

        self.assertEqual(len(part.predicted_annotations), 2)

        self.assertEqual(part.predicted_annotations[0].text, 'c.A100G')
        self.assertEqual(part.predicted_annotations[0].offset, 10)

        self.assertEqual(part.predicted_annotations[1].text, 'p.V100Q')
        self.assertEqual(part.predicted_annotations[1].offset, 18)


if __name__ == '__main__':
    unittest.main()