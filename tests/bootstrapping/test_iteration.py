from unittest import TestCase
from nala.bootstrapping.iteration import Iteration
from nose.plugins.attrib import attr


@attr('slow')
class TestIteration(TestCase):
    def test_learning(self):
        iteration = Iteration('tmpbootstrapping')
        iteration.learning()
        iteration.docselection(nr=10)
        print(len(iteration.candidates.documents))
        iteration.tagging()
        print("\n\n\n\n\n\nPREDICTION\n\n\n\n\n")
        for ann in iteration.candidates.predicted_annotations():
            print(ann)
