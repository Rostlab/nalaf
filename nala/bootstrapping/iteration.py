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
from nala.utils.writers import TagTogFormat


class Iteration(Cacheable):
    def __init__(self, folder=None, iteration_nr=None):
        super().__init__()

        if folder is not None:
            self.bootstrapping_folder = os.path.abspath(folder)
        else:
            self.bootstrapping_folder = os.path.abspath("resources/bootstrapping")

        # represents the iteration
        self.number = -1

        # empty init variables
        self.train = None  # first
        self.candidates = None  # non predicted docselected
        self.predicted = None  # predicted docselected
        self.crf = CRFSuite("/usr/local/Cellar/crfsuite/0.12")

        # todo discussion on config file in bootstrapping root or iteration_n check for n

        if iteration_nr is None:
            # find iteration number
            _iteration_name = self.bootstrapping_folder + "/iteration_*/"
            for fn in glob.glob(_iteration_name):
                match = re.search(r'/iteration_(\d+)/$', fn)
                found_iteration = int(match.group(1))
                if found_iteration > self.number:
                    self.number = found_iteration

            self.number = self.number + 1  # NOTE before last iteration now current iteration
        else:
            self.number = iteration_nr
        # current folders
        self.current_folder = os.path.join(self.bootstrapping_folder, "iteration_{}".format(self.number))
        self.candidates_folder = os.path.join(self.current_folder, 'candidates')
        self.reviewed_folder = os.path.join(self.current_folder, 'reviewed')

    # def get_previous_ids(self):
    #     for dpath, dname, fname in os.walk(self.bootstrapping_folder):
    #         if fname:
    # todo this method get previous ids
    #             pass

    def learning(self):
        """
        Learning: base + iterations 1..n-1
        :return:
        """
        print_verbose("\n\n\n======Learning======\n\n\n")
        # parse base + reviewed files
        # base
        base_folder = os.path.join(self.bootstrapping_folder, "iteration_0/base/")
        html_base_folder = base_folder + "html/"
        annjson_base_folder = base_folder + "annjson/"
        self.train = HTMLReader(html_base_folder).read()
        AnnJsonAnnotationReader(annjson_base_folder).annotate(self.train)

        # extend for each next iteration
        if self.number > 0:
            for i in range(1, self.number):
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
        print_verbose("\n\n\n======DocSelection======\n\n\n")
        self.candidates = generate_documents(nr)

    def tagging(self):
        # tagging
        print_verbose("\n\n\n======Tagging======\n\n\n")
        PrepareDatasetPipeline().execute(self.candidates)
        self.crf.create_input_file(self.candidates, 'predict')
        self.crf.tag('-m default_model -i predict > output.txt')
        self.crf.read_predictions(self.candidates)

        # export candidates to candidates folder
        os.mkdir(self.current_folder)

        ttf_candidates = TagTogFormat(self.candidates, self.candidates_folder)
        ttf_candidates.export_html()
        ttf_candidates.export_ann_json(0.99)  # 0.99 for beginning

    def manual_review_import(self):
        """
        Parse from iteration_n/reviewed folder in anndoc format.
        :return:
        """
        # todo consolidate into one dataset
        self.reviewed = HTMLReader(os.path.join(self.candidates_folder, 'html')).read()
        AnnJsonAnnotationReader(os.path.join(self.candidates_folder, 'annjson'), delete_incomplete_docs=False).annotate(
            self.reviewed)
        AnnJsonAnnotationReader(os.path.join(self.reviewed_folder)).annotate(self.reviewed)
        for ann in self.reviewed.annotations():
            print(ann)

        print("now predicted annotations")

        for ann in self.reviewed.predicted_annotations():
            print(ann)

        # automatic evaluation

    def evaluation(self):
        """
        When Candidates and Reviewed are existing do automatic evaluation and calculate performances
        :return:
        """
        pass