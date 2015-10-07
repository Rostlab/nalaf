import abc
import os
import re


class PMIDFilter:
    """
    Abstract base class for filtering out a list of PMIDs (pubmed IDs)
    according to some criterion.

    The logic behind whether a PMID is filtered out or kept should not depend on
    the text of the article, only on some external source. If your logic depends on
    the article text associated with the PMID subclass DocumentFilter instead.


    Subclasses that inherit this class should:
    * Be named [Name]PMIDFilter
    * Implement the abstract method filter as a generator
    meaning that if some criterion is fulfilled then [yield] that pmid

    When implementing filter first iterate for each PMID then apply logic to allow chaining of filters.
    """
    @abc.abstractmethod
    def filter(self, pmids):
        """
        :type pmids: collections.Iterable[str]
        """
        pass


class AlreadyConsideredPMIDFilter(PMIDFilter):
    """
    Filters out PMIDs that we have considered in a previous iteration of the bootstrapping procedure.
    """
    def __init__(self, bootstrapping_root, iteration_n):
        self.bootstrapping_root = bootstrapping_root
        """the root directory containing all iterations of the bootstrapping"""
        self.iteration_n = iteration_n
        """the number of the current iteration"""

    def filter(self, pmids):
        considered_pmids = set()
        # find all previously considered PMIDs
        for root, sub_folders, files in os.walk(self.bootstrapping_root):
            # find the iteration number if it exists
            which_iteration = re.search('iteration_([0-9]+)', root)
            # if it doesen't set it to a value we won't use
            which_iteration = int(which_iteration.group(1)) if which_iteration else self.iteration_n
            if which_iteration < self.iteration_n and files:
                for file in files:
                    # try to find the pmid based on file name convention
                    file_match = re.search('-((PMC)?([0-9]+))\.(ann|plain)?\.(json|html)', file)
                    # extract the pmid if you found it
                    if file_match:
                        considered_pmids.add(file_match.group(1))
                    else:
                        considered_pmids.add(os.path.basename(file))

        # yield only pmids that have not been considered in a previous iteration
        for pmid in pmids:
            if pmid not in considered_pmids:
                yield pmid


