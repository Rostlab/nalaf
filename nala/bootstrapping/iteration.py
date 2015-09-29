import glob
import os
import re
from nala.bootstrapping.pmid_filters import AlreadyConsideredPMIDFilter
from nala.learning.postprocessing import PostProcessing
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
from nala.preprocessing.definers import ExclusiveNLDefiner
import pkg_resources


class Iteration():
    def __init__(self, folder=None, iteration_nr=None, crfsuite_path=None):
        super().__init__()

        if folder is not None:
            self.bootstrapping_folder = os.path.abspath(folder)
        else:
            self.bootstrapping_folder = os.path.abspath("resources/bootstrapping")

        if crfsuite_path is None:
            crfsuite_path = os.path.abspath(r'crfsuite')
        else:
            crfsuite_path = os.path.abspath(crfsuite_path)

        # represents the iteration
        self.number = -1

        # stats file
        self.stats_file = os.path.join(self.bootstrapping_folder, 'stats.csv')

        # empty init variables
        self.train = None  # first
        self.candidates = None  # non predicted docselected
        self.predicted = None  # predicted docselected
        self.crf = CRFSuite(crfsuite_path)

        # discussion on config file in bootstrapping root or iteration_n check for n
        # note currently using parameter .. i think that s the most suitable

        if iteration_nr is None:
            # find iteration number
            _iteration_name = self.bootstrapping_folder + "/iteration_*/"
            for fn in glob.glob(_iteration_name):
                match = re.search(r'/iteration_(\d+)/$', fn)
                found_iteration = int(match.group(1))
                if found_iteration > self.number:
                    self.number = found_iteration

            # check for candidates and reviewed
            if os.path.isdir(os.path.join(self.bootstrapping_folder, "iteration_{}".format(self.number), 'candidates')):
                if os.path.isdir(os.path.join(self.bootstrapping_folder, "iteration_{}".format(self.number), 'reviewed')):
                    # todo check for evaluation done (writing in csv file)
                    self.number += 1
            if self.number == 0:
                self.number += 1
        else:
            self.number = iteration_nr
        # current folders
        self.current_folder = os.path.join(self.bootstrapping_folder, "iteration_{}".format(self.number))
        self.candidates_folder = os.path.join(self.current_folder, 'candidates')
        self.reviewed_folder = os.path.join(self.current_folder, 'reviewed')

    def before_annotation(self, nr_new_docs=10):
        self.learning()
        self.docselection(nr=nr_new_docs)
        self.tagging()

    def after_annotation(self):
        self.manual_review_import()
        self.evaluation()

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
        print(len(self.train.documents))
        # extend for each next iteration
        if self.number > 1:
            for i in range(1, self.number):
                # get new dataset
                path_to_read = os.path.join(self.bootstrapping_folder, "iteration_{}".format(i))
                tmp_data = HTMLReader(path_to_read + "/candidates/html/").read()
                AnnJsonAnnotationReader(path_to_read + "/reviewed/").annotate(tmp_data)

                # extend learning_data
                # todo has to be tested
                self.train.extend_dataset(tmp_data)
        # prune parts without annotations
        self.train.prune()

        # generate features etc.
        PrepareDatasetPipeline().execute(self.train)
        BIEOLabeler().label(self.train)
        print(len(self.train.documents))
        # crfsuite part
        self.crf.create_input_file(self.train, 'train')
        self.crf.learn()
        # todo save model to iteration_0 folder as bin_model

    def docselection(self, nr=2):
        """
        Does the same as generate_documents(n) but the bootstrapping folder is specified in here.
        :param nr: amount of new documents wanted
        """
        print_verbose("\n\n\n======DocSelection======\n\n\n")
        from nala.structures.data import Dataset
        from nala.structures.pipelines import DocumentSelectorPipeline
        from itertools import count
        c = count(1)

        dataset = Dataset()
        with DocumentSelectorPipeline(pmid_filters=[AlreadyConsideredPMIDFilter(self.bootstrapping_folder, self.number)]) as dsp:
            for pmid, document in dsp.execute():
                dataset.documents[pmid] = document
                # if we have generated enough documents stop
                if next(c) == nr:
                    break
        self.candidates = dataset

    def tagging(self, threshold_val=0.99):
        # tagging
        print_verbose("\n\n\n======Tagging======\n\n\n")
        PrepareDatasetPipeline().execute(self.candidates)
        self.crf.create_input_file(self.candidates, 'predict')
        self.crf.tag('-m default_model -i predict > output.txt')
        self.crf.read_predictions(self.candidates)
        PostProcessing().process(self.candidates)

        # export candidates to candidates folder
        os.mkdir(self.current_folder)

        ttf_candidates = TagTogFormat(self.candidates, self.candidates_folder)
        ttf_candidates.export_html()
        ttf_candidates.export_ann_json(threshold_val)  # 0.99 for beginning

    # todo divide into 2 parts with 1st being learning, docselection, tagging and the 2nd being manual import and evaluation

    def manual_review_import(self):
        """
        Parse from iteration_n/reviewed folder in anndoc format.
        :return:
        """
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
        ExclusiveNLDefiner().define(self.reviewed)
        results = MentionLevelEvaluator().evaluate(self.reviewed)
        # print(results)
        with open(self.stats_file, 'a') as f:
            f.write("{}\t{}\n".format(self.number, "\t".join(list(str(r) for r in results))))
