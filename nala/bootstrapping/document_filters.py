import abc


class DocumentFilter:
    """
    Abstract base class for filtering out a list of documents (given as a Dataset object)
    according to some criterion.

    Subclasses that inherit this class should:
    * Be named [Name]DocumentFilter
    * Implement the abstract method filter as a generator
    meaning that if some criterion is fulfilled then [yield] that document

    When implementing filter first iterate for each PMID then apply logic to allow chaining of filters.
    """

    @abc.abstractmethod
    def filter(self, documents):
        """
        :type documents: collections.Iterable[nala.structures.data.Document]
        """
        pass


class KeywordsDocumentFilter(DocumentFilter):
    """
    Filters our documents that do not contain any of the given keywords in any of their parts.
    """
    def __init__(self, keywords=None):
        if not keywords:
            keywords = ('mutation', 'variation', 'substitution', 'insertion', 'deletion', 'snp')
        self.keywords = keywords
        """the keywords which the document should contain"""

    def filter(self, documents):
        """
        :type documents: collections.Iterable[nala.structures.data.Document]
        """
        for doc in documents:
            # if any part of the document contains any of the keywords
            # yield that document
            if any(any(keyword in part.text.lower() for keyword in self.keywords)
                   for part in doc.parts.values()):
                yield doc


