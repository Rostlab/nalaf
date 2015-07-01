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
        expected_length = iter([13, 8])
        expected_nr = iter([0, 4])
        expected_nr_up = iter([1, 0])
        expected_nr_lo = iter([12, 3])
        expected_nr_alpha = iter([13, 3])
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

        for token in self.dataset.tokens():
            self.assertEqual(token.features['length[0]'], next(expected_length))
            self.assertEqual(token.features['num_nr[0]'], next(expected_nr))
            self.assertEqual(token.features['num_up[0]'], next(expected_nr_up))
            self.assertEqual(token.features['num_lo[0]'], next(expected_nr_lo))
            self.assertEqual(token.features['num_alpha[0]'], next(expected_nr_alpha))
            self.assertEqual(token.features['num_spec_chars[0]'], next(expected_nr_spec_chars),
                             msg="word={} | feature={}".format(token.word, token.features['num_spec_chars[0]']))
            # print(token.features['num_has_chr_key[0]'], token.word)
            self.assertEqual(token.features['num_has_chr_key[0]'], next(expected_chr_key))


            import json
            print(json.dumps(token.features, indent=3, sort_keys=True))

    def test_mutation_article_bp(self):
        mutat_article = ""  # NOTE is this programming ok?

        self.assertEqual(self.feature.mutation_article_bp("three"), "Base")
        self.assertEqual(self.feature.mutation_article_bp("BLUSDmb"), "Byte")
        self.assertEqual(self.feature.mutation_article_bp("1232bp"), "bp")
        self.assertEqual(self.feature.mutation_article_bp("the"), None)

    def type1(self, str):
        if self.reg_type1.match(str):
            return "Type1"
        elif self.reg_type12.match(str):
            return "Type1_2"
        else:
            return None

    def type2(self, str):
        return "Type2" if str == "p" else None

    def dna_symbols(self, str):
        return "DNASym" if self.reg_dna_symbols.match(str) else None

    def protein_symbols(self, str):
        uc_tmp = str  # upper case
        lc_tmp = str.lower()  # lower case

        if self.reg_prot_symbols1.match(lc_tmp):
            return "ProteinSymFull"
        elif self.reg_prot_symbols2.match(lc_tmp):
            return "ProteinSymTri"
        # TODO last token include: "&& $last_token[...]"
        elif self.reg_prot_symbols3.match(lc_tmp):
            return "ProteinSymTriSub"
        elif self.reg_prot_symbols4.match(uc_tmp):
            return "ProteinSymChar"
        else:
            return None

    def rscode(self, str):
        if self.reg_rs_code1.match(str):
            return "RSCode"
        elif self.reg_rs_code2.match(str):
            return "RSCode"
        else:
            return None

if __name__ == '__main__':
    unittest.main()
