from nala.preprocessing.spliters import Splitter, NLTKSplitter
from nala.preprocessing.tokenizers import Tokenizer, TmVarTokenizer
from nala.features import FeatureGenerator
from nala.features.simple import SimpleFeatureGenerator
from nala.features.stemming import PorterStemFeatureGenerator
from nala.features.tmvar import TmVarFeatureGenerator, TmVarDictionaryFeatureGenerator
from nala.features.window import WindowFeatureGenerator
from nala.utils.cache import Cacheable
from nala.bootstrapping.document_filters import DocumentFilter
from nala.bootstrapping.document_filters import KeywordsDocumentFilter, HighRecallRegexDocumentFilter
from nala.bootstrapping.pmid_filters import PMIDFilter
from nala.bootstrapping.pmid_filters import AlreadyConsideredPMIDFilter
from nala.bootstrapping.utils import DownloadArticle
from nala.bootstrapping.utils import UniprotDocumentSelector, DownloadArticle


class PrepareDatasetPipeline:
    """
    Prepares an instance of a dataset by executing modules in fixed order.
        * First executes the sentence splitter
        * Next executes the tokenizer
        * Finally executes each feature generator in the order they were provided

    :type splitter: nala.structures.data.Splitter
    :param splitter: the module responsible for splitting the text into sentences
    :type tokenizer: nala.structures.data.Tokenizer
    :param tokenizer: the module responsible for splitting the sentences into tokens
    :type feature_generators: collections.Iterable[FeatureGenerator]
    :param feature_generators: one or more modules responsible for generating features
    """

    def __init__(self, splitter=None, tokenizer=None, feature_generators=None):
        if not splitter:
            splitter = NLTKSplitter()
        if not tokenizer:
            tokenizer = TmVarTokenizer()
        if not feature_generators:
            include = ['pattern0[0]', 'pattern1[0]', 'pattern2[0]', 'pattern3[0]', 'pattern4[0]', 'pattern5[0]',
                       'pattern6[0]', 'pattern7[0]', 'pattern8[0]', 'pattern9[0]', 'pattern10[0]', 'word[0]', 'stem[0]']
            feature_generators = [SimpleFeatureGenerator(), PorterStemFeatureGenerator(), TmVarFeatureGenerator(),
                                  TmVarDictionaryFeatureGenerator(),
                                  WindowFeatureGenerator(template=(-3, -2, -1, 1, 2, 3), include_list=include)]

        if isinstance(splitter, Splitter):
            self.splitter = splitter
        else:
            raise TypeError('not an instance that implements Splitter')

        if isinstance(tokenizer, Tokenizer):
            self.tokenizer = tokenizer
        else:
            raise TypeError('not an instance that implements Tokenizer')

        if hasattr(feature_generators, '__iter__'):
            for index, feature_generator in enumerate(feature_generators):
                if not isinstance(feature_generator, FeatureGenerator):
                    raise TypeError('not an instance that implements FeatureGenerator at index {}'.format(index))
            self.feature_generators = feature_generators
        elif isinstance(feature_generators, FeatureGenerator):
            self.feature_generators = [feature_generators]
        else:
            raise TypeError('not an instance or iterable of instances that implements FeatureGenerator')

    def execute(self, dataset):
        """
        :type dataset: nala.structures.data.Dataset()
        """
        self.splitter.split(dataset)
        self.tokenizer.tokenize(dataset)
        for feature_generator in self.feature_generators:
            feature_generator.generate(dataset)


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
