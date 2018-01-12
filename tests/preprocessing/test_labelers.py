import unittest
from nalaf.structures.data import Dataset, Document, Part, Token, Entity
from nalaf.utils.readers import StringReader
from nalaf.preprocessing.spliters import NLTKSplitter
from nalaf.preprocessing.tokenizers import TmVarTokenizer
from nalaf.preprocessing.labelers import BIOLabeler, BIEOLabeler, TmVarLabeler

STUB_ENTITY_CLASS_ID = 'e_x'


class TestLabelers(unittest.TestCase):
    def setUp(self):
        self.dataset = StringReader('some text ... (c.2708_2711delTTAG, p.V903GfsX905) ... text').read()
        NLTKSplitter().split(self.dataset)
        TmVarTokenizer().tokenize(self.dataset)
        part = list(self.dataset.parts())[0]
        part.annotations.append(Entity(STUB_ENTITY_CLASS_ID, 15, 'c.2708_2711delTTAG'))
        part.annotations.append(Entity(STUB_ENTITY_CLASS_ID, 35, 'p.V903GfsX905'))

    def test_bio_labeler(self):
        BIOLabeler().label(self.dataset)
        labels = [token.original_labels[0].value for token in self.dataset.tokens()]
        expected = ['O', 'O', 'O', 'O', 'O', 'O', 'B-e_x', 'I-e_x', 'I-e_x', 'I-e_x', 'I-e_x', 'I-e_x', 'I-e_x', 'O',
                    'B-e_x', 'I-e_x', 'I-e_x', 'I-e_x', 'I-e_x', 'I-e_x', 'I-e_x', 'O', 'O', 'O', 'O', 'O']
        self.assertEqual(labels, expected)

    def test_bieo_labeler(self):
        BIEOLabeler().label(self.dataset)
        labels = [token.original_labels[0].value for token in self.dataset.tokens()]
        expected = ['O', 'O', 'O', 'O', 'O', 'O', 'B-e_x', 'I-e_x', 'I-e_x', 'I-e_x', 'I-e_x', 'I-e_x', 'E-e_x', 'O',
                    'B-e_x', 'I-e_x', 'I-e_x', 'I-e_x', 'I-e_x', 'I-e_x', 'E-e_x', 'O', 'O', 'O', 'O', 'O']
        self.assertEqual(labels, expected)

    def test_tmvar_labeler(self):
        TmVarLabeler(STUB_ENTITY_CLASS_ID).label(self.dataset)
        labels = [token.original_labels[0].value for token in self.dataset.tokens()]
        expected = ['O', 'O', 'O', 'O', 'O', 'O', 'A', 'I', 'P', 'P', 'P', 'T', 'W', 'O',
                    'A', 'I', 'W', 'P', 'I', 'M', 'P', 'O', 'O', 'O', 'O', 'O']
        self.assertEqual(labels, expected)


if __name__ == '__main__':
    unittest.main()
