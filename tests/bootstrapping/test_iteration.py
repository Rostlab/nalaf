from unittest import TestCase
from nala.bootstrapping.iteration import Iteration
from nose.plugins.attrib import attr
from nala.utils.writers import TagTogFormat
import os
from utils.readers import HTMLReader


@attr('slow')
class TestIteration(TestCase):
    def test_learning(self):
        iteration = Iteration('tmpbootstrapping', iteration_nr=1)
        # iteration.learning()
        # iteration.docselection(nr=10)
        # print(len(iteration.candidates.documents))
        # iteration.tagging()
        # print("\n\n\n\n\n\nPREDICTION\n\n\n\n\n")
        # for ann in iteration.candidates.predicted_annotations():
        #     print(ann)
        # note current working directory changes after crfsuite to crfsuite-folder
        # ttf = TagTogFormat(iteration.candidates, "/Users/carsten/tmp")
        # ttf.export_html()
        # ttf.export_ann_json(0.7)

        iteration.manual_review_import()
        iteration.evaluation()
