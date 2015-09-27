import unittest
from nose.plugins.attrib import attr
from nala.bootstrapping.utils import generate_documents
from nala.bootstrapping.document_filters import HighRecallRegexDocumentFilter
from nala.structures.data import Dataset


@attr('slow')
class TestHighRecallRegexFilter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    def test_filter(self):
        generate_documents(1)

if __name__ == '__main__':
    unittest.main()