import unittest
from nala.structures.data import Dataset, Document, Part, Token, Label


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
        self.dataset.form_predicted_annotations()

        part = self.dataset.documents['doc_1'].parts['p1']

        self.assertEqual(len(part.predicted_annotations), 2)

        self.assertEqual(part.predicted_annotations[0].text, 'c.A100G')
        self.assertEqual(part.predicted_annotations[0].offset, 10)

        self.assertEqual(part.predicted_annotations[1].text, 'p.V100Q')
        self.assertEqual(part.predicted_annotations[1].offset, 18)



