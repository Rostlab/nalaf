from unittest import TestCase
from nala.bootstrapping.iteration import Iteration
from nose.plugins.attrib import attr
import os


@attr('slow')
class TestIteration(TestCase):
    def test_learning(self):
        iteration = Iteration(iteration_nr=1, crfsuite_path=r'crfsuite')
        # iteration.before_annotation(nr_new_docs=5)
        # iteration.after_annotation()

        # iteration.learning()
        # iteration.docselection(nr=5)
        # print(len(iteration.candidates.documents))
        # iteration.tagging()
        # print("\n\n\n\n\n\nPREDICTION\n\n\n\n\n")
        # for ann in iteration.candidates.predicted_annotations():
        #     print(ann)

        iteration.manual_review_import()
        iteration.evaluation()
