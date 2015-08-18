from unittest import TestCase
from nala.structures.data import Dataset, Document, Part, Token, Label
from nala.utils import MUT_CLASS_ID


class TestDataset(TestCase):
    def test_parts(self):
        self.fail()

    def test_annotations(self):
        self.fail()

    def test_predicted_annotations(self):
        self.fail()

    def test_sentences(self):
        self.fail()

    def test_tokens(self):
        self.fail()

    def test_partids_with_parts(self):
        self.fail()

    def test_annotations_with_partids(self):
        self.fail()

    def test_all_annotations_with_ids(self):
        self.fail()

    def test_form_predicted_annotations(self):
        self.fail()

    def test_clean_nl_definitions(self):
        self.fail()

    def test_stats(self):
        self.fail()


class TestDocument(TestCase):
    def test_key_value_parts(self):
        self.fail()

    def test_get_unique_mentions(self):
        self.fail()

    def test_get_size(self):
        self.fail()


class TestToken(TestCase):
    def test_init(self):
        self.fail()

    def test_repr(self):
        self.fail()


class TestFeatureDictionary(TestCase):
    def test_setitem(self):
        self.fail()


class TestAnnotation(TestCase):
    def test_init(self):
        self.fail()

    def test_repr(self):
        self.fail()

    def test_eq(self):
        self.fail()


class TestLabel(TestCase):
    def test_repr(self):
        self.fail()

    def test_init(self):
        self.fail()


class TestPart(TestCase):
    def test_init(self):
        self.fail()

    def test_iter(self):
        self.fail()


class TestMentionLevel(TestCase):
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
