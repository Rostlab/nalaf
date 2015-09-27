from unittest import TestCase
from nala.bootstrapping.iteration import Iteration

__author__ = 'carsten'


class TestIteration(TestCase):
    def test_learning(self):
        iteration = Iteration('tmpbootstrapping')
        # iteration.learning()
        iteration.docselection(nr=10)
        print(len(iteration.candidates.documents))
        # iteration.tagging()
        # print("run again")
        # for ann in iteration.candidates.predicted_annotations():
        #     print(ann)
