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
    def filter(self, dataset):
        """
        :type dataset: nala.structures.data.Dataset
        """
        pass
