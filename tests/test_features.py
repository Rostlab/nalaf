import unittest
from nala.structures.data import Dataset, Document, Part, Token
from nala.features.tmvar import TmVarDefault
from nala.features import FeatureGenerator


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

        cls.feature = TmVarDefault()
        cls.feature.generate(dataset=cls.dataset)

    def test_implements_feature_interface(self):
        self.assertIsInstance(self.feature, FeatureGenerator)

    def test_feature_objects_are_created(self):
        for token in self.dataset.tokens():
            self.assertTrue(len(token.features) > 0)

    def test_generate(self):
        expected_nr = iter([0, 4])
        expected_nr_up = iter([1, 0])
        expected_nr_lo = iter(["L:4+", 3])
        expected_nr_alpha = iter(["A:4+", 3])
        expected_nr_spec_chars = iter([None, "SpecC1"])
        expected_chr_key = iter(["ChroKey", "ChroKey"])
        expected_mutat_type = iter(["MutatWord", None])
        expected_mutat_word = iter(["FrameShiftType", "MutatType"])
        # expected_mutat_article
        # expected_type1
        # expected_type2
        # expected_dna_symbols
        # expected_protein_symbols
        # expected_rscode
        # NOTE implemented as extra features

        for token in self.dataset.tokens():
            self.assertEqual(token.features['num_nr[0]'], next(expected_nr))
            self.assertEqual(token.features['num_up[0]'], next(expected_nr_up))
            self.assertEqual(token.features['num_lo[0]'], next(expected_nr_lo))
            self.assertEqual(token.features['num_alpha[0]'], next(expected_nr_alpha))
            self.assertEqual(token.features['num_spec_chars[0]'], next(expected_nr_spec_chars),
                             msg="word={} | feature={}".format(token.word, token.features['num_spec_chars[0]']))
            # print(token.features['num_has_chr_key[0]'], token.word)
            self.assertEqual(token.features['num_has_chr_key[0]'], next(expected_chr_key))

            # import json
            # print(json.dumps(token.features, indent=3, sort_keys=True))

    def test_mutation_article_bp(self):
        self.assertEqual(self.feature.mutation_article_bp("three"), "Base")
        self.assertEqual(self.feature.mutation_article_bp("BLUSDmb"), "Byte")
        self.assertEqual(self.feature.mutation_article_bp("Flowerpowerbp"), "bp")
        self.assertEqual(self.feature.mutation_article_bp("the"), None)

    def test_type1(self):
        self.assertEqual(self.feature.type1("g"), "Type1")
        self.assertEqual(self.feature.type1("orf"), "Type1_2")
        self.assertEqual(self.feature.type1("blaaa"), None)

    def test_type2(self):
        self.assertEqual(self.feature.type2("p"), "Type2")
        self.assertEqual(self.feature.type2("as"), None)

    def test_dna_symbols(self):
        self.assertEqual(self.feature.dna_symbols("A"), "DNASym")
        self.assertEqual(self.feature.dna_symbols("asd"), None)

    def test_protein_symbols(self):
        self.assertEqual(self.feature.protein_symbols("glutamine"), "ProteinSymFull")
        self.assertEqual(self.feature.protein_symbols("asn"), "ProteinSymTri")
        self.assertEqual(self.feature.protein_symbols("eu"), "ProteinSymTriSub")
        self.assertEqual(self.feature.protein_symbols("X"), "ProteinSymChar")
        self.assertEqual(self.feature.protein_symbols("flowerpower"), None)

    def test_rscode(self):
        self.assertEqual(self.feature.rscode("rs0"), "RSCode")
        self.assertEqual(self.feature.rscode("rs"), "RSCode")
        self.assertEqual(self.feature.rscode("rsssss"), None)

if __name__ == '__main__':
    unittest.main()
