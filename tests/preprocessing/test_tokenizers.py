import unittest
from nalaf.preprocessing.tokenizers import Tokenizer
from nalaf.preprocessing.tokenizers import NLTK_TOKENIZER, TmVarTokenizer
from nalaf.structures.data import Dataset, Document, Part, Token


class Test_NLTK_TOKENIZER(unittest.TestCase):
    """
    Test the NLTK_TOKENIZER class and it's main method tokenize()
    """

    @classmethod
    def setup_class(cls):
        # create a sample dataset to test
        cls.dataset = Dataset()

        doc_id1 = Document()
        # 15 tokens in 2 sentences
        doc_id1.parts['p1'] = Part('This is some sample text. This is another, sample sentence with coma.')
        doc_id1.parts['p1'].sentences_ = ['This is some sample text.', 'This is another, sample sentence with coma.']

        cls.dataset.documents['doc_id1'] = doc_id1

        cls.tokenizer = NLTK_TOKENIZER
        cls.tokenizer.tokenize(cls.dataset)

    def test_implements_tokenizer_interface(self):
        self.assertIsInstance(self.tokenizer, Tokenizer)

    def test_token_objects_are_created(self):
        for token in self.dataset.tokens():
            self.assertIsInstance(token, Token)

    def test_number_of_tokens_as_expected(self):
        self.assertEqual(len(list(self.dataset.tokens())), 15)

    def test_tokens_as_expected(self):
        expected = iter(['This', 'is', 'some', 'sample', 'text', '.',
                         'This', 'is', 'another', ',', 'sample', 'sentence', 'with', 'coma', '.'])
        for token in self.dataset.tokens():
            self.assertEqual(token.word, next(expected))


class TestTmVarTokenizer(unittest.TestCase):
    """
    Test the TmVarTokenizer class and it's main method tokenize()
    """

    @classmethod
    def setup_class(cls):
        # create a sample dataset to test
        cls.dataset = Dataset()

        doc_id1 = Document()
        # 15 tokens in 2 sentences
        doc_id1.parts['p1'] = Part('this is some sample text. it contains this c.2708_2711delTTAG mutation.')
        doc_id1.parts['p1'].sentences_ = ['this is some sample text.', 'it contains this c.2708_2711delTTAG mutation.']

        cls.dataset.documents['doc_id1'] = doc_id1

        cls.tokenizer = TmVarTokenizer()
        cls.tokenizer.tokenize(cls.dataset)

    def test_implements_tokenizer_interface(self):
        self.assertIsInstance(self.tokenizer, Tokenizer)

    def test_token_objects_are_created(self):
        for token in self.dataset.tokens():
            self.assertIsInstance(token, Token)

    def test_number_of_tokens_as_expected(self):
        self.assertEqual(len(list(self.dataset.tokens())), 18)

    def test_tokens_as_expected(self):
        expected = iter(['this', 'is', 'some', 'sample', 'text', '.',
                         'it', 'contains', 'this', 'c', '.', '2708', '_', '2711', 'del', 'TTAG', 'mutation', '.'])
        for token in self.dataset.tokens():
            self.assertEqual(token.word, next(expected))


if __name__ == '__main__':
    unittest.main()
