import unittest
from bootstrapping.document_filters import HighRecallRegexDocumentFilter
from structures.data import Dataset


class TestHighRecallRegexFilter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    def test_filter(self):
        from nala.bootstrapping import UniprotDocumentSelector
        from nala.bootstrapping.document_filters import KeywordsDocumentFilter
        from nala.bootstrapping.pmid_filters import AlreadyConsideredPMIDFilter
        from nala.bootstrapping import DownloadArticle
        from itertools import count
        c = count(1)

        dataset = Dataset()

        max_docs = 5

        # use in context manager to enable caching
        with UniprotDocumentSelector() as uds:
            with DownloadArticle() as da:
                for pmid, document in \
                        HighRecallRegexDocumentFilter().filter(
                            KeywordsDocumentFilter().filter(
                                da.download(
                                    AlreadyConsideredPMIDFilter('resources/bootstrapping', 4).filter(
                                        uds.get_pubmed_ids()))), log_nr=max_docs):
                    dataset.documents[pmid] = document
                    # if we have generated enough documents stop
                    if next(c) == max_docs:
                        break

if __name__ == '__main__':
    unittest.main()