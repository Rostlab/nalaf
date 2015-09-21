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

        # use in context manager to enable caching
        with UniprotDocumentSelector() as uds:
            for pmid, document in \
                    HighRecallRegexDocumentFilter.filter(KeywordsDocumentFilter().filter(
                        DownloadArticle().download(
                            AlreadyConsideredPMIDFilter('idp4_pmid_list.txt', 4).filter(
                                uds.get_pubmed_ids())))):
                dataset.documents[pmid] = document

                # if we have generated enough documents stop
                if next(c) == 5:
                    break
        print(str(dataset))

if __name__ == '__main__':
    unittest.main()