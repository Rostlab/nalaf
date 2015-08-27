import unittest
from nala.structures.data import Dataset, Document, Part, Token, Annotation
from nala.preprocessing.labelers import BIOLabeler, BIEOLabeler, TmVarLabeler
from nala.utils import MUT_CLASS_ID


class TestLabelers(unittest.TestCase):
    def setUp(self):
        self.dataset = Dataset()
        doc = Document()
        part = Part('some text ... (c.2708_2711delTTAG, p.V903GfsX905) ... text')
        self.dataset.documents['doc_1'] = doc
        doc.parts['part_1'] = part
        from nala.preprocessing.spliters import NLTKSplitter
        from nala.preprocessing.tokenizers import TmVarTokenizer

        NLTKSplitter().split(self.dataset)
        TmVarTokenizer().tokenize(self.dataset)
        # part.sentences = [[Token('some'), Token('text'), Token('.'), Token('.'), Token('.'), Token('('), Token('c'),
        #                    Token('.'), Token('2708'), Token('_'), Token('2711'), Token('del'), Token('TTAG'),
        #                    Token(','), Token('p'), Token('.'), Token('V'), Token('903'), Token('G'), Token('fs'),
        #                    Token('X'), Token('905'), Token(')'), Token('.'), Token('.'), Token('.'), Token('text')]]

        part.annotations.append(Annotation(MUT_CLASS_ID, 15, 'c.2708_2711delTTAG'))
        part.annotations.append(Annotation(MUT_CLASS_ID, 35, 'p.V903GfsX905'))

    def test_bio_labeler(self):
        BIOLabeler().label(self.dataset)
        labels = [token.original_labels[0].value for token in self.dataset.tokens()]
        expected = ['O', 'O', 'O', 'O', 'O', 'O', 'B-e_2', 'I-e_2', 'I-e_2', 'I-e_2', 'I-e_2', 'I-e_2', 'I-e_2', 'O',
                    'B-e_2', 'I-e_2', 'I-e_2', 'I-e_2', 'I-e_2', 'I-e_2', 'I-e_2', 'I-e_2', 'O', 'O', 'O', 'O', 'O']
        self.assertEqual(labels, expected)

    def test_bieo_labeler(self):
        BIEOLabeler().label(self.dataset)
        labels = [token.original_labels[0].value for token in self.dataset.tokens()]
        expected = ['O', 'O', 'O', 'O', 'O', 'O', 'B-e_2', 'I-e_2', 'I-e_2', 'I-e_2', 'I-e_2', 'I-e_2', 'E-e_2', 'O',
                    'B-e_2', 'I-e_2', 'I-e_2', 'I-e_2', 'I-e_2', 'I-e_2', 'I-e_2', 'E-e_2', 'O', 'O', 'O', 'O', 'O']
        self.assertEqual(labels, expected)

    def test_tmvar_labeler(self):
        TmVarLabeler().label(self.dataset)
        labels = [token.original_labels[0].value for token in self.dataset.tokens()]
        expected = ['O', 'O', 'O', 'O', 'O', 'O', 'A', 'I', 'P', 'I', 'P', 'T', 'M', 'O',
                    'A', 'I', 'W', 'P', 'M', 'F', 'F', 'S', 'O', 'O', 'O', 'O', 'O']
        self.assertEqual(labels, expected)


if __name__ == '__main__':
    unittest.main()
