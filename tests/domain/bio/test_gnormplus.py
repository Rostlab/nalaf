import unittest
from nose.plugins.attrib import attr
from nalaf.learning.taggers import StubSameSentenceRelationExtractor
from nalaf.structures.data import *
from nalaf.learning.domain.bio import GNormPlusGeneTagger
from nalaf.preprocessing.spliters import NLTKSplitter
from nalaf.preprocessing.tokenizers import TmVarTokenizer


@attr('slow')
class TestGNormPlusGeneTagger(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = Dataset()

        doc = Document()
        docid = '15878741'

        title = Part("Identification of novel mutations of the human N-acetylglutamate synthase gene and their functional investigation by expression studies.")
        abstract = Part("The mitochondrial enzyme N-acetylglutamate synthase (NAGS) produces N-acetylglutamate serving as an allosteric activator of carbamylphosphate synthetase 1, the first enzyme of the urea cycle. Autosomal recessively inherited NAGS deficiency (NAGSD) leads to severe neonatal or late-onset hyperammonemia. To date few patients have been described and the gene involved was described only recently. In this study, another three families affected by NAGSD were analyzed for NAGS gene mutations resulting in the identification of three novel missense mutations (C200R [c.598T > C], S410P [c.1228T > C], A518T [c.1552G > A]). In order to investigate the effects of these three and two additional previously published missense mutations on enzyme activity, the mutated proteins were overexpressed in a bacterial expression system using the NAGS deficient E. coli strain NK5992. All mutated proteins showed a severe decrease in enzyme activity providing evidence for the disease-causing nature of the mutations. In addition, we expressed the full-length NAGS wild type protein including the mitochondrial leading sequence, the mature protein as well as a highly conserved core protein. NAGS activity was detected in all three recombinant proteins but varied regarding activity levels and response to stimulation by l-arginine. In conclusion, overexpression of wild type and mutated NAGS proteins in E. coli provides a suitable tool for functional analysis of NAGS deficiency.")

        ann_mut1 = Entity(class_id=MUT_CLASS_ID, offset=18, text='enzyme')
        ann_mut2 = Entity(class_id=MUT_CLASS_ID, offset=192, text='Autosomal')
        abstract.annotations += [ann_mut1, ann_mut2]

        doc.parts['title'] = title
        doc.parts['abstract'] = abstract
        cls.data.documents[docid] = doc

    def test_tag(self):
        # todo question is that the proper way? with predicts_classes
        GNormPlusGeneTagger().tag(self.data, uniprot=True)
        NLTKSplitter().split(self.data)
        TmVarTokenizer().tokenize(self.data)
        StubSameSentenceRelationExtractor(PRO_CLASS_ID, MUT_CLASS_ID, PRO_REL_MUT_CLASS_ID).annotate(self.data)
        self.assertEqual(len([x for x in self.data.annotations() if x.class_id == PRO_CLASS_ID]), 0)
        self.assertEqual(len([x for x in self.data.annotations() if x.class_id == MUT_CLASS_ID]), 2)
        self.assertEqual(len([x for x in self.data.relations() if x.class_id == PRO_REL_MUT_CLASS_ID]), 0)
        self.data.purge_false_relationships()
        self.assertEqual(len([x for x in self.data.relations() if x.class_id == PRO_REL_MUT_CLASS_ID]), 0)
        del self.data.documents['15878741'].parts['abstract'].annotations[0]
        self.assertEqual(len([x for x in self.data.annotations() if x.class_id == MUT_CLASS_ID]), 1)
        self.data.purge_false_relationships()
        self.assertEqual(len([x for x in self.data.relations() if x.class_id == PRO_REL_MUT_CLASS_ID]), 0)

if __name__ == '__main__':
    unittest.main()
