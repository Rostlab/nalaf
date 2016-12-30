import unittest
from nalaf.structures.data import Dataset, Document, Part, Token, Label, Entity
from nalaf.preprocessing.spliters import NLTKSplitter
from nalaf import print_verbose, print_debug
from nalaf.preprocessing.tokenizers import Tokenizer, NLTK_TOKENIZER, GenericTokenizer
from nalaf.features import get_spacy_nlp_english
from nalaf.preprocessing.parsers import Parser, SpacyParser
import sys


STUB_ENTITY_CLASS_ID_1 = 'e_x_1'
STUB_ENTITY_CLASS_ID_2 = 'e_x_2'
STUB_RELATION_CLASS_ID_2 = 'r_x'

TEST_SENTENCES_SINGLE_ROOT = [
    "Arabidopsis cotyledon - specific chloroplast biogenesis factor CYO1 is a protein disulfide isomerase .",
    "FKBP12-rapamycin target TOR2 is a vacuolar protein with an associated phosphatidylinositol-4 kinase activity .",
    "TMEM59 was found to be a ubiquitously expressed , Golgi - localized protein .",
    "This indicates that Mdv1p possesses a Dnm1p - independent mitochondrial targeting signal .",
    "Dnm1p - independent targeting of Mdv1p to mitochondria requires MDV2 .",
    "The activated ROP11 recruits MIDD1 to induce local disassembly of cortical microtubules .",
    "Conversely , cortical microtubules eliminate active ROP11 from the plasma membrane through MIDD1 .",
    "GOLPH3L antagonizes GOLPH3 to determine Golgi morphology .",
    "HERC2 coordinates ubiquitin - dependent assembly of DNA repair factors on damaged chromosomes .",
    "Pivotal role of AtSUVH2 in heterochromatic histone methylation and gene silencing in Arabidopsis .",
    "PHAX and CRM1 are required sequentially to transport U3 snoRNA to nucleoli .",
    "CpSufE activates the cysteine desulfurase CpNifS for chloroplastic Fe - S cluster formation .",
    "YMR313c/TGL3 encodes a novel triacylglycerol lipase located in lipid particles of Saccharomyces cerevisiae .",
    "However , overexpression of ATG21 leads to CPY secretion .",
    "PP2A colocalizes with shugoshin at centromeres and is required for centromeric protection .",
]

TEST_SENTENCES_MULTI_ROOT = [
    # SS
    "Import assays with pea ( Pisum sativum ) chloroplasts showed that PyrR and PyrD are taken up and proteolytically processed ."
]

class TestFloydWarshall(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dataset = Dataset()

        doc1 = Document()
        cls.dataset.documents['TEST_SENTENCES_SINGLE_ROOT'] = doc1

        for s in TEST_SENTENCES_SINGLE_ROOT:
            part = Part(s)
            doc1.parts[s] = part

        doc2 = Document()
        cls.dataset.documents['TEST_SENTENCES_MULTI_ROOT'] = doc2

        for s in TEST_SENTENCES_MULTI_ROOT:
            part = Part(s)
            doc2.parts[s] = part

        cls.nlp = get_spacy_nlp_english(load_parser=True)
        cls.parser = SpacyParser(cls.nlp)
        cls.splitter = NLTKSplitter()
        cls.tokenizer = GenericTokenizer(lambda string: (tok.text for tok in cls.nlp.tokenizer(string)))

        cls.splitter.split(cls.dataset)
        cls.tokenizer.tokenize(cls.dataset)
        cls.parser.parse(cls.dataset)


    def test_distance_u_u_is_0(self):
        pass

    def test_distance_u_v_is_v_u(self):
        pass

    def test_path_u_v_is_reverseof_v_u(self):
        pass


if __name__ == '__main__':
    unittest.main()
