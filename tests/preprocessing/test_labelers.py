import unittest
from nala.structures.data import Dataset, Document, Part, Token, Entity
from nala.utils.readers import StringReader
from nala.preprocessing.spliters import NLTKSplitter
from nala.preprocessing.tokenizers import TmVarTokenizer
from nala.preprocessing.labelers import BIOLabeler, BIEOLabeler, TmVarLabeler
from nala.utils import MUT_CLASS_ID


class TestLabelers(unittest.TestCase):
    def setUp(self):
        self.dataset = StringReader('some text ... (c.2708_2711delTTAG, p.V903GfsX905) ... text').read()
        NLTKSplitter().split(self.dataset)
        TmVarTokenizer().tokenize(self.dataset)
        part = list(self.dataset.parts())[0]
        part.annotations.append(Entity(MUT_CLASS_ID, 15, 'c.2708_2711delTTAG'))
        part.annotations.append(Entity(MUT_CLASS_ID, 35, 'p.V903GfsX905'))

    def test_bio_labeler(self):
        BIOLabeler().label(self.dataset)
        labels = [token.original_labels[0].value for token in self.dataset.tokens()]
        expected = ['O', 'O', 'O', 'O', 'O', 'O', 'B-e_2', 'I-e_2', 'I-e_2', 'I-e_2', 'I-e_2', 'I-e_2', 'I-e_2', 'O',
                    'B-e_2', 'I-e_2', 'I-e_2', 'I-e_2', 'I-e_2', 'I-e_2', 'I-e_2', 'O', 'O', 'O', 'O', 'O']
        self.assertEqual(labels, expected)

    def test_bieo_labeler(self):
        BIEOLabeler().label(self.dataset)
        labels = [token.original_labels[0].value for token in self.dataset.tokens()]
        expected = ['O', 'O', 'O', 'O', 'O', 'O', 'B-e_2', 'I-e_2', 'I-e_2', 'I-e_2', 'I-e_2', 'I-e_2', 'E-e_2', 'O',
                    'B-e_2', 'I-e_2', 'I-e_2', 'I-e_2', 'I-e_2', 'I-e_2', 'E-e_2', 'O', 'O', 'O', 'O', 'O']
        self.assertEqual(labels, expected)

    def test_tmvar_labeler(self):
        TmVarLabeler().label(self.dataset)
        labels = [token.original_labels[0].value for token in self.dataset.tokens()]
        expected = ['O', 'O', 'O', 'O', 'O', 'O', 'A', 'I', 'P', 'P', 'P', 'T', 'W', 'O',
                    'A', 'I', 'W', 'P', 'I', 'M', 'P', 'O', 'O', 'O', 'O', 'O']
        self.assertEqual(labels, expected)


if __name__ == '__main__':
    unittest.main()
