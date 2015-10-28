import configparser
from unittest import TestCase
from nala.learning.taggers import GNormPlusGeneTagger, StubSameSentenceRelationExtractor
from nala.bootstrapping.iteration import Iteration
from nose.plugins.attrib import attr
import os
from nala.utils import PRO_REL_MUT_CLASS_ID, UNIPROT_ID, ENTREZ_GENE_ID
from nala.utils.writers import TagTogFormat
import argparse


@attr('slow')
class TestIteration(TestCase):
    def test_learning(self):
        # config = configparser.ConfigParser()

        # config.read_file('config.ini')
        #
        # html_path = config.get('paths', 'html_path')
        # ann_path = config.get('paths', 'ann_path')
        # crf_path = config.get('paths', 'crf_path')
        # bstrap_path = config.get('paths', 'bstrap_path')


        # iteration = Iteration(crfsuite_path=os.path.abspath('crfsuite'), iteration_nr=2)
        iteration = Iteration(crfsuite_path=os.path.abspath('crfsuite'), iteration_nr=3)
        # iteration.before_annotation(nr_new_docs=10)
        # iteration.after_annotation()

        # iteration.cross_validation(5)

        # iteration.learning()
        # iteration.docselection(nr=5)
        # print(len(iteration.candidates.documents))
        # iteration.tagging()
        # print("\n\n\n\n\n\nPREDICTION\n\n\n\n\n")
        # for ann in iteration.candidates.predicted_annotations():
        #     print(ann)

        # iteration.manual_review_import()
        # iteration.evaluation()
        pass

    def test_init(self):
        iteration = Iteration(iteration_nr=1, crfsuite_path=r'crfsuite')
        # self.assertEqual(iteration.number, 2)
        iteration.manual_review_import()
        # print(iteration.reviewed)
        GNormPlusGeneTagger().tag(iteration.reviewed, uniprot=True)
        StubSameSentenceRelationExtractor().tag(iteration.reviewed)
        TagTogFormat(iteration.reviewed, to_save_to='flowers').export(0.8)
