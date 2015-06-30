import unittest
from nala.preprocessing.tokenizers import Tokenizer
from nala.preprocessing.tokenizers import NLTKTokenizer, TmVarTokenizer
from nala.structures.data import Dataset, Document, Part, Token
from nala.features.tmvar import TmVarDefault
from nala.features import FeatureGenerator

class TmVarDefaultTest(unittest.TestCase):
    """
    Test the NLTKTokenizer class and it's main method tokenize()
    """
    @classmethod
    def setUpClass(cls):
        # create a sample dataset to test
        cls.dataset = Dataset()

        doc_id1 = Document()
        # 15 tokens in 2 sentences
        doc_id1.parts['p1'] = Part('one ')
        doc_id1.parts['p1'].sentences = [[Token('oneA'), Token('j23.23')]]
        cls.dataset.documents['doc_id1'] = doc_id1

        cls.feature = TmVarDefault()
        cls.feature.generate(dataset=cls.dataset)


    def test_implements_feature_interface(self):
        self.assertIsInstance(self.feature, FeatureGenerator)

    def test_feature_objects_are_created(self):
        for token in self.dataset.tokens():
            self.assertTrue(len(token.features) > 0)

    def test_feature_attributes(self):
        expected_length = iter([4, 6])
        expected_nr = iter([0,4])
        expected_nr_up = iter([1,0])
        expected_nr_lo = iter([3,1])
        expected_nr_alpha = iter([4,1])
        expected_nr_spec_chars = iter([None, "SpecC1"])
        expected_chr_key = iter([None, None])

        for token in self.dataset.tokens():
            self.assertEqual(token.features['length[0]'], next(expected_length))
            self.assertEqual(token.features['num_nr[0]'], next(expected_nr))
            self.assertEqual(token.features['num_up[0]'], next(expected_nr_up))
            self.assertEqual(token.features['num_lo[0]'], next(expected_nr_lo))
            self.assertEqual(token.features['num_alpha[0]'], next(expected_nr_alpha))
            self.assertEqual(token.features['num_spec_chars[0]'], next(expected_nr_spec_chars), msg=token.word)
            self.assertEqual(token.features['num_has_chr_key[0]'], next(expected_chr_key))

            # print(token.features)

if __name__ == '__main__':
    unittest.main()