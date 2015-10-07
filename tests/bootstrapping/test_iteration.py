from unittest import TestCase
from learning.taggers import GNormPlusGeneTagger, RelationshipExtractionGeneMutation
from nala.bootstrapping.iteration import Iteration
from nose.plugins.attrib import attr
import os
from utils import PRO_REL_MUT_CLASS_ID, UNIPROT_ID, ENTREZ_GENE_ID
from utils.writers import TagTogFormat


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

        # iteration.manual_review_import()
        # iteration.evaluation()

    def test_init(self):
        iteration = Iteration(iteration_nr=1, crfsuite_path=r'crfsuite')
        # self.assertEqual(iteration.number, 2)
        iteration.manual_review_import()
        # print(iteration.reviewed)
        GNormPlusGeneTagger([PRO_REL_MUT_CLASS_ID]).tag(iteration.reviewed, uniprot=True)
        RelationshipExtractionGeneMutation([UNIPROT_ID, ENTREZ_GENE_ID]).tag(iteration.reviewed)
        TagTogFormat(iteration.reviewed, to_save_to='flowers').export(0.8)
