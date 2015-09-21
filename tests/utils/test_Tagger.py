from unittest import TestCase
from nala.utils.tagger import TmVarTagger

__author__ = 'carst'


class TestTmVarTagger(TestCase):
    def test_generate_abstracts(self):
        pmids = ['12559908']

        data = TmVarTagger().generate_abstracts(pmids)

        print(data)
        for docid in data.documents:
            print(data.documents[docid])
