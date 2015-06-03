import unittest
from nala.preprocessing.spliters import NLTKSplitter
from nala.preprocessing.tokenizers import Tokenizer
from nala.preprocessing.tokenizers import NLTKTokenizer
from nala.structures.data import Dataset, Document, Part, Token


class TestNLTKTokenizer(unittest.TestCase):
    """
    Test the NLTKTokenizer class and it's main method tokenize()
    """
    @classmethod
    def setup_class(cls):
        # create a sample dataset to test
        cls.dataset = Dataset()

        doc_id1 = Document()
        # 15 tokens in 2 senteces
        doc_id1.parts['p1'] = Part('This is some sample text. This is another, sample sentence with coma.')
        cls.dataset.documents['doc_id1'] = doc_id1

        NLTKSplitter().split(cls.dataset)
        cls.tokenizer = NLTKTokenizer()
        cls.tokenizer.tokenize(cls.dataset)

    def test_implements_tokenizer_interface(self):
        self.assertIsInstance(self.tokenizer, Tokenizer)

    def test_token_objects_are_created(self):
        for token in self.dataset.tokens():
            self.assertIsInstance(token, Token)

    def test_number_of_toknes_as_expected(self):
        self.assertEqual(len(list(self.dataset.tokens())), 15)

    def test_tokens_as_expected(self):
        expected = iter(['This', 'is', 'some', 'sample', 'text', '.',
                         'This', 'is', 'another', ',', 'sample', 'sentence', 'with', 'coma', '.'])
        for token in self.dataset.tokens():
            self.assertEqual(token.word, next(expected))
