import unittest
from nala.structures.data import Dataset, Document, Part, Token, FeatureDictionary
from nala.features.tmvar import TmVarFeatureGenerator
from nala.features import FeatureGenerator
from nala.features import eval_binary_feature
from nala.features.tmvar import TmVarDictionaryFeatureGenerator
from nala.utils.readers import StringReader
from nala.preprocessing.spliters import NLTKSplitter
from nala.preprocessing.tokenizers import TmVarTokenizer


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
        doc_id1.parts['p1'] = Part('insertionefsA dup23.23')
        doc_id1.parts['p1'].sentences = [[Token('insertionefsA', 0), Token('dup23.23', 14)]]
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
        expected_nr_spec_chars = iter(["NoSpecC", "SpecC1"])
        # NOTE implemented as extra features

        for token in self.dataset.tokens():
            self.assertEqual(token.features['num_nr[0]'], next(expected_nr))
            self.assertEqual(token.features['num_up[0]'], next(expected_nr_up))
            self.assertEqual(token.features['num_lo[0]'], next(expected_nr_lo))
            self.assertEqual(token.features['num_alpha[0]'], next(expected_nr_alpha))
            self.assertEqual(token.features['num_spec_chars[0]'], next(expected_nr_spec_chars),
                             msg="word={} | feature={}".format(token.word, token.features['num_spec_chars[0]']))

    # OPTIONAL implement separate test functions for each feature that is already implemented in test_generate

    def test_mutation_type(self):
        self.assertEqual(self.feature.mutation_type("fs"), "FrameShiftType")
        self.assertEqual(self.feature.mutation_type("del"), "MutatType")
        self.assertEqual(self.feature.mutation_type("der"), None)

    def test_mutation_word(self):
        feature_dic = FeatureDictionary()
        eval_binary_feature(feature_dic, 'mutat_word', self.feature.reg_mutat_word.match, 'repeats')
        self.assertEqual(feature_dic.get('mutat_word[0]'), 1)

        feature_dic = FeatureDictionary()
        eval_binary_feature(feature_dic, 'mutat_word', self.feature.reg_mutat_word.match, 'repssts')
        self.assertEqual(feature_dic.get('mutat_word[0]'), None)

    def test_mutation_article_bp(self):
        self.assertEqual(self.feature.mutation_article_bp("three"), "Base")
        self.assertEqual(self.feature.mutation_article_bp("BLUSDmb"), "Byte")
        self.assertEqual(self.feature.mutation_article_bp("Flowerpowerbp"), "bp")
        self.assertEqual(self.feature.mutation_article_bp("the"), "NoMutArticle")

    def test_type1(self):
        self.assertEqual(self.feature.is_special_type_1("g"), "Type1")
        self.assertEqual(self.feature.is_special_type_1("orf"), "Type1_2")
        self.assertEqual(self.feature.is_special_type_1("blaaa"), "NotSpecType1")

    def test_type2(self):
        feature_dic = FeatureDictionary()
        eval_binary_feature(feature_dic, 'type2', lambda x: x == 'p', 'p')
        self.assertEqual(feature_dic.get('type2[0]'), 1)

        feature_dic = FeatureDictionary()
        eval_binary_feature(feature_dic, 'type2', lambda x: x == 'p', 'as')
        self.assertEqual(feature_dic.get('type2[0'), None)

    def test_dna_symbols(self):
        feature_dic = FeatureDictionary()
        eval_binary_feature(feature_dic, 'dna_symbols', self.feature.reg_dna_symbols.match, 'A')
        self.assertEqual(feature_dic.get('dna_symbols[0]'), 1)

        feature_dic = FeatureDictionary()
        eval_binary_feature(feature_dic, 'dna_symbols', self.feature.reg_dna_symbols.match, 'asd')
        self.assertEqual(feature_dic.get('dna_symbols[0]'), None)

    def test_protein_symbols(self):
        self.assertEqual(self.feature.has_protein_symbols("glutamine", "bla"), "ProteinSymFull")
        self.assertEqual(self.feature.has_protein_symbols("asn", "bla"), "ProteinSymTri")
        self.assertEqual(self.feature.has_protein_symbols("eu", "X"), "ProteinSymTriSub")
        self.assertEqual(self.feature.has_protein_symbols("X", "X"), "ProteinSymChar")
        self.assertEqual(self.feature.has_protein_symbols("flowerpower", "AAA"), "NoProteinSymbol")

    def test_rscode(self):
        self.assertEqual(self.feature.has_rscode("rs0"), "RSCode")
        self.assertEqual(self.feature.has_rscode("rs"), "RSCode")
        self.assertEqual(self.feature.has_rscode("rsssss"), "NoRSCode")

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
    def test_generate_patterns_026(self):
        dataset = StringReader('token c.2708_2711delTTAG token').read()
        NLTKSplitter().split(dataset)
        TmVarTokenizer().tokenize(dataset)
        TmVarDictionaryFeatureGenerator().generate(dataset)

        token_features = [{key: value for key, value in token.features.items() if value is not 'O'}
                          for token in dataset.tokens()]
        self.assertEqual(token_features[0], {})
        self.assertEqual(token_features[1], {'pattern2[0]': 'B', 'pattern0[0]': 'B'})
        self.assertEqual(token_features[2], {'pattern2[0]': 'I', 'pattern0[0]': 'I'})
        self.assertEqual(token_features[3], {'pattern2[0]': 'I', 'pattern0[0]': 'I'})
        self.assertEqual(token_features[4], {'pattern2[0]': 'I', 'pattern0[0]': 'I'})
        self.assertEqual(token_features[5], {'pattern2[0]': 'E', 'pattern6[0]': 'B', 'pattern0[0]': 'I'})
        self.assertEqual(token_features[6], {'pattern6[0]': 'I', 'pattern0[0]': 'I'})
        self.assertEqual(token_features[7], {'pattern6[0]': 'E', 'pattern0[0]': 'E'})
        self.assertEqual(token_features[8], {})

    def test_generate_patterns_136(self):
        dataset = StringReader('token IVS2-58_55insT token').read()
        NLTKSplitter().split(dataset)
        TmVarTokenizer().tokenize(dataset)
        TmVarDictionaryFeatureGenerator().generate(dataset)

        token_features = [{key: value for key, value in token.features.items() if value is not 'O'}
                          for token in dataset.tokens()]
        self.assertEqual(token_features[0], {})
        self.assertEqual(token_features[1], {'pattern3[0]': 'B', 'pattern1[0]': 'B'})
        self.assertEqual(token_features[2], {'pattern3[0]': 'I', 'pattern1[0]': 'I'})
        self.assertEqual(token_features[3], {'pattern3[0]': 'I', 'pattern1[0]': 'I'})
        self.assertEqual(token_features[4], {'pattern3[0]': 'I', 'pattern1[0]': 'I'})
        self.assertEqual(token_features[5], {'pattern3[0]': 'I', 'pattern1[0]': 'I'})
        self.assertEqual(token_features[6], {'pattern3[0]': 'E', 'pattern1[0]': 'I', 'pattern6[0]': 'B'})
        self.assertEqual(token_features[7], {'pattern1[0]': 'I', 'pattern6[0]': 'I'})
        self.assertEqual(token_features[8], {'pattern1[0]': 'E', 'pattern6[0]': 'E'})
        self.assertEqual(token_features[9], {})

    def test_generate_patterns_245(self):
        dataset = StringReader('token c.A436C token').read()
        NLTKSplitter().split(dataset)
        TmVarTokenizer().tokenize(dataset)
        TmVarDictionaryFeatureGenerator().generate(dataset)

        token_features = [{key: value for key, value in token.features.items() if value is not 'O'}
                          for token in dataset.tokens()]
        self.assertEqual(token_features[0], {})
        self.assertEqual(token_features[1], {'pattern4[0]': 'B', 'pattern2[0]': 'B'})
        self.assertEqual(token_features[2], {'pattern4[0]': 'I', 'pattern2[0]': 'I'})
        self.assertEqual(token_features[3], {'pattern4[0]': 'I', 'pattern2[0]': 'I', 'pattern5[0]': 'B'})
        self.assertEqual(token_features[4], {'pattern4[0]': 'I', 'pattern2[0]': 'I', 'pattern5[0]': 'I'})
        self.assertEqual(token_features[5], {'pattern4[0]': 'E', 'pattern2[0]': 'I', 'pattern5[0]': 'E'})
        self.assertEqual(token_features[6], {})

    def test_generate_patterns_789(self):
        dataset = StringReader('token p.G204VfsX28 token').read()
        NLTKSplitter().split(dataset)
        TmVarTokenizer().tokenize(dataset)
        TmVarDictionaryFeatureGenerator().generate(dataset)

        token_features = [{key: value for key, value in token.features.items() if value is not 'O'}
                          for token in dataset.tokens()]
        self.assertEqual(token_features[0], {})
        self.assertEqual(token_features[1], {'pattern7[0]': 'B', 'pattern9[0]': 'B', 'pattern8[0]': 'B'})
        self.assertEqual(token_features[2], {'pattern7[0]': 'I', 'pattern9[0]': 'I', 'pattern8[0]': 'I'})
        self.assertEqual(token_features[3], {'pattern7[0]': 'I', 'pattern9[0]': 'I', 'pattern8[0]': 'I'})
        self.assertEqual(token_features[4], {'pattern7[0]': 'I', 'pattern9[0]': 'I', 'pattern8[0]': 'I'})
        self.assertEqual(token_features[5], {'pattern7[0]': 'I', 'pattern9[0]': 'E'})
        self.assertEqual(token_features[6], {'pattern7[0]': 'I'})
        self.assertEqual(token_features[7], {'pattern7[0]': 'I'})
        self.assertEqual(token_features[8], {})

    def test_patterns(self):
        fg = TmVarDictionaryFeatureGenerator()
        self.assertTrue(fg.patterns[0].match('c.2708_2711delTTAG'))
        self.assertTrue(fg.patterns[1].match('IVS2-58_55insT'))
        self.assertTrue(fg.patterns[2].match('c.467C>A'))
        self.assertTrue(fg.patterns[3].match('IVS3+18C>T '))
        self.assertTrue(fg.patterns[4].match('c.A436C'))
        self.assertTrue(fg.patterns[5].match('A436C'))
        self.assertTrue(fg.patterns[6].match('912delTA'))
        self.assertTrue(fg.patterns[7].match('p.G204VfsX28'))
        self.assertTrue(fg.patterns[8].match('p.G204V'))
        self.assertTrue(fg.patterns[9].match('p.Ser157Ser'))
        self.assertTrue(fg.patterns[10].match('p.Ser119fsX'))
if __name__ == '__main__':
    unittest.main()
