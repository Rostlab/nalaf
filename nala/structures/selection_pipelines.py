from nala.bootstrapping.document_filters import DocumentFilter, KeywordsDocumentFilter, HighRecallRegexDocumentFilter
from nala.bootstrapping.pmid_filters import PMIDFilter, AlreadyConsideredPMIDFilter
from nala.bootstrapping.utils import UniprotDocumentSelector, DownloadArticle
from nala.utils.cache import Cacheable

__author__ = 'Aleksandar'


class DocumentSelectorPipeline:
    """
    A document selection pipeline used for the purposes of bootstrapping that executes
    a series of generator modules in specific fixed order:
        * First executes the initial document selector
        * Next applies a series of pmid filters
        * Then it transforms the stream of pmids into a stream of documents
        * Finally applies a series of document filters

    :type initial_document_selector: nala.bootstrapping.utils.UniprotDocumentSelector
    :type article_downloader: nala.bootstrapping.utils.DownloadArticle
    :type pmid_filters: collections.Iterable[nala.bootstrapping.pmid_filters.PMIDFilter]
    :param pmid_filters: one or more generator modules responsible for filtering pmids
    :type document_filters: collections.Iterable[nala.bootstrapping.document_filters.DocumentFilter]
    :param document_filters: one or more generator modules responsible for filtering documents
    """
    def __init__(self, pmid_filters=None, document_filters=None):
        self.initial_document_selector = UniprotDocumentSelector()
        self.article_downloader = DownloadArticle()

        if pmid_filters:
            # check the type of the provided pmid filter
            if hasattr(pmid_filters, '__iter__'):
                for index, pmid_filter in enumerate(pmid_filters):
                    if not isinstance(pmid_filter, PMIDFilter):
                        raise TypeError('not an instance that implements PMIDFilter at index {}'.format(index))
                self.pmid_filters = pmid_filters
            elif isinstance(pmid_filters, PMIDFilter):
                self.pmid_filters = [pmid_filters]
            else:
                raise TypeError('not an instance that implements PMIDFilter')
        else:
            self.pmid_filters = [AlreadyConsideredPMIDFilter('resources/bootstrapping', 4)]

        if document_filters:
            # check the type of the provided document filters
            if hasattr(document_filters, '__iter__'):
                for index, document_filter in enumerate(document_filters):
                    if not isinstance(document_filter, DocumentFilter):
                        raise TypeError('not an instance that implements DocumentFilter at index {}'.format(index))
                self.document_filters = document_filters
            elif isinstance(document_filters, DocumentFilter):
                self.document_filters = [document_filters]
            else:
                raise TypeError('not an instance that implements DocumentFilter')
        else:
            self.document_filters = [KeywordsDocumentFilter(), HighRecallRegexDocumentFilter()]

    def __enter__(self):
        if isinstance(self.initial_document_selector, Cacheable):
            self.initial_document_selector.__enter__()
        if isinstance(self.article_downloader, Cacheable):
            self.article_downloader.__enter__()
        for pmid_filter in self.pmid_filters:
            if isinstance(pmid_filter, Cacheable):
                pmid_filter.__enter__()
        for document_filer in self.document_filters:
            if isinstance(document_filer, Cacheable):
                document_filer.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if isinstance(self.initial_document_selector, Cacheable):
            self.initial_document_selector.__exit__(exc_type, exc_val, exc_tb)
        if isinstance(self.article_downloader, Cacheable):
            self.article_downloader.__exit__(exc_type, exc_val, exc_tb)
        for pmid_filter in self.pmid_filters:
            if isinstance(pmid_filter, Cacheable):
                pmid_filter.__exit__(exc_type, exc_val, exc_tb)
        for document_filer in self.document_filters:
            if isinstance(document_filer, Cacheable):
                document_filer.__exit__(exc_type, exc_val, exc_tb)

    def execute(self):
        # initialized the pipeline
        pipeline = self.initial_document_selector.get_pubmed_ids()

        # chain all the pmid filters
        for pmid_filter in self.pmid_filters:
            pipeline = pmid_filter.filter(pipeline)

        # convert pmids to documents
        pipeline = self.article_downloader.download(pipeline)

        # chain all the document filters
        for document_filter in self.document_filters:
            pipeline = document_filter.filter(pipeline)

        return pipeline