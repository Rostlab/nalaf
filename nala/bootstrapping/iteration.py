import glob
import os
import re
from nala import print_verbose
from nala.learning.crfsuite import CRFSuite
from nala.structures.pipelines import PrepareDatasetPipeline
from nala.utils.annotation_readers import AnnJsonAnnotationReader
from nala.utils.cache import Cacheable
from nala.utils.readers import HTMLReader
from nala.preprocessing.labelers import BIEOLabeler
from nala.learning.evaluators import MentionLevelEvaluator
from nala.bootstrapping.utils import generate_documents


class Iteration(Cacheable):
    def __init__(self, folder=None):
        super().__init__()

        if folder is not None:
            self.bootstrapping_folder = folder
        else:
            self.bootstrapping_folder = "resources/bootstrapping"

        # represents the iteration
        self.number = -1

        # empty init variables
        self.train = None  # first
        self.candidates = None  # non predicted docselected
        self.predicted = None  # predicted docselected
        self.crf = CRFSuite("/usr/local/Cellar/crfsuite/0.12")

        # todo discussion on config file in bootstrapping root or iteration_n check for n

        # find iteration number
        _iteration_name = self.bootstrapping_folder + "/iteration_*/"
        for fn in glob.glob(_iteration_name):
            match = re.search(r'/iteration_(\d+)/$', fn)
            found_iteration = int(match.group(1))
            if found_iteration > self.number:
                self.number = found_iteration

    def learning(self):
        """
        Learning: base + iterations 1..n-1
        :return:
        """
        print_verbose("Learning")
        # parse base + reviewed files
        # base
        base_folder = os.path.join(self.bootstrapping_folder, "iteration_0/base/")
        html_base_folder = base_folder + "html/"
        annjson_base_folder = base_folder + "annjson/"
        self.train = HTMLReader(html_base_folder).read()
        AnnJsonAnnotationReader(annjson_base_folder).annotate(self.train)

        # extend for each next iteration
        if self.number > 0:
            for i in range(1, self.number + 1):
                # get new dataset
                path_to_read = os.path.join(self.bootstrapping_folder, "iteration_{}".format(i), "reviewed/")
                tmp_data = HTMLReader(path_to_read + "html/").read()
                AnnJsonAnnotationReader(path_to_read + "annjson/").annotate(tmp_data)

                # extend learning_data
                # todo has to be tested
                self.train.extend_dataset(tmp_data)

        # generate features etc.
        PrepareDatasetPipeline().execute(self.train)
        BIEOLabeler().label(self.train)

        # crfsuite part
        self.crf.create_input_file(self.train, 'train')
        self.crf.learn()

    def docselection(self, nr=2):
        # docselection
        print_verbose("DocSelection")
        self.candidates = generate_documents(nr)

    def tagging(self):
        # tagging
        print_verbose("Tagging")
        PrepareDatasetPipeline().execute(self.candidates)
        self.crf.create_input_file(self.candidates, 'predict')
        self.crf.tag('-m default_model -i predict > output.txt')
        self.crf.read_predictions(self.candidates)

        # manual review

        # automatic evaluation