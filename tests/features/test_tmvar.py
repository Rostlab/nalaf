import unittest
from nala.structures.data import Dataset, Document, Part, Token
from nala.features.tmvar import TmVarFeatureGenerator
from nala.features import FeatureGenerator
import re


class TmVarDefaultTest(unittest.TestCase):
    """
    Test the TmVarDefault class and it's main method generate() with including separate helper-functions for each feature that gets generated.
    """

    @classmethod
    def setUpClass(cls):
        # create a sample dataset to test
        cls.dataset = Dataset()

        doc_id1 = Document()
        # 15 tokens in 2 sentences
        doc_id1.parts['p1'] = Part('one ')
        doc_id1.parts['p1'].sentences = [[Token('insertionefsA'), Token('dup23.23')]]
        cls.dataset.documents['doc_id1'] = doc_id1

        cls.feature = TmVarFeatureGenerator()
        cls.feature.generate(dataset=cls.dataset)

    def test_implements_feature_interface(self):
        self.assertIsInstance(self.feature, FeatureGenerator)

    def test_feature_objects_are_created(self):
        for token in self.dataset.tokens():
            self.assertTrue(len(token.features) > 0)

    def test_generate(self):
        expected_nr = iter([0, 4])
        expected_nr_up = iter([1, 0])
        expected_nr_lo = iter(["4+", 3])
        expected_nr_alpha = iter(["4+", 3])
        expected_nr_spec_chars = iter([None, "SpecC1"])
        expected_chr_key = iter(["ChroKey", "ChroKey"])
        # NOTE implemented as extra features

        for token in self.dataset.tokens():
            self.assertEqual(token.features['num_nr[0]'], next(expected_nr))
            self.assertEqual(token.features['num_up[0]'], next(expected_nr_up))
            self.assertEqual(token.features['num_lo[0]'], next(expected_nr_lo))
            self.assertEqual(token.features['num_alpha[0]'], next(expected_nr_alpha))
            self.assertEqual(token.features['num_spec_chars[0]'], next(expected_nr_spec_chars),
                             msg="word={} | feature={}".format(token.word, token.features['num_spec_chars[0]']))
            self.assertEqual(token.features['num_has_chr_key[0]'], next(expected_chr_key))

    # OPTIONAL implement separate test functions for each feature that is already implemented in test_generate

    def test_mutation_type(self):
        self.assertEqual(self.feature.mutation_type("fs"), "FrameShiftType")
        self.assertEqual(self.feature.mutation_type("del"), "MutatType")
        self.assertEqual(self.feature.mutation_type("der"), None)

    def test_mutation_word(self):
        self.assertEqual(self.feature.mutation_word("repeats"), "MutatWord")
        self.assertEqual(self.feature.mutation_word("repessts"), None)

    def test_mutation_article_bp(self):
        self.assertEqual(self.feature.mutation_article_bp("three"), "Base")
        self.assertEqual(self.feature.mutation_article_bp("BLUSDmb"), "Byte")
        self.assertEqual(self.feature.mutation_article_bp("Flowerpowerbp"), "bp")
        self.assertEqual(self.feature.mutation_article_bp("the"), None)

    def test_type1(self):
        self.assertEqual(self.feature.is_special_type_1("g"), "Type1")
        self.assertEqual(self.feature.is_special_type_1("orf"), "Type1_2")
        self.assertEqual(self.feature.is_special_type_1("blaaa"), None)

    def test_type2(self):
        self.assertEqual(self.feature.is_special_type_2("p"), "Type2")
        self.assertEqual(self.feature.is_special_type_2("as"), None)

    def test_dna_symbols(self):
        self.assertEqual(self.feature.has_dna_symbols("A"), "DNASym")
        self.assertEqual(self.feature.has_dna_symbols("asd"), None)

    def test_protein_symbols(self):
        self.assertEqual(self.feature.has_protein_symbols("glutamine", "bla"), "ProteinSymFull")
        self.assertEqual(self.feature.has_protein_symbols("asn", "bla"), "ProteinSymTri")
        self.assertEqual(self.feature.has_protein_symbols("eu", "X"), "ProteinSymTriSub")
        self.assertEqual(self.feature.has_protein_symbols("X", "X"), "ProteinSymChar")
        self.assertEqual(self.feature.has_protein_symbols("flowerpower", "AAA"), None)

    def test_rscode(self):
        self.assertEqual(self.feature.has_rscode("rs0"), "RSCode")
        self.assertEqual(self.feature.has_rscode("rs"), "RSCode")
        self.assertEqual(self.feature.has_rscode("rsssss"), None)

    def test_shape1(self):
        self.assertEqual(self.feature.word_shape_1("Bs0ssaDB2"), "Aa0aaaAA0")

    def test_shape2(self):
        self.assertEqual(self.feature.word_shape_2("Bs0ssaDB2"), "aa0aaaaa0")

    def test_shape3(self):
        self.assertEqual(self.feature.word_shape_3("Bs0ssaDB2"), "Aa0aA0")

    def test_shape4(self):
        self.assertEqual(self.feature.word_shape_4("Bs0ssaDB2"), "a0a0")

    def test_prefix_pattern(self):
        self.assertEqual(self.feature.prefix_pattern("A"), ["A", None, None, None, None])
        self.assertEqual(self.feature.prefix_pattern("ASDASD"), ["A", "AS", "ASD", "ASDA", "ASDAS"])

    def test_suffix_pattern(self):
        self.assertEqual(self.feature.suffix_pattern("ABC"), ["C", "BC", "ABC", None, None])


class TestTmVarDictionaryFeatureGenerator(unittest.TestCase):
    def test_init(self):
        self.fail()  # TODO

    def test_generate(self):
        self.fail()  # TODO


class TestWindowFeatureGenerator(unittest.TestCase):
    def test_init(self):
        self.fail()  # TODO

    def test_generate(self):
        self.fail()  # TODO


if __name__ == '__main__':
    unittest.main()
