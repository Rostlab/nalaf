from nala.utils.readers import HTMLReader
from nala.preprocessing.spliters import NLTKSplitter
from nala.preprocessing.tokenizers import NLTKTokenizer
from nala.preprocessing.annotators import ReadFromAnnJsonAnnotator#
from nala.preprocessing.labelers import SimpleLabeler
from nala.preprocessing.definers import TmVarRegexNLDefiner
from nala.features.simple import SimpleFeatureGenerator
from nala.utils.crfsuite import CRFSuite

if __name__ == "__main__":

    html_path = r'C:\Users\Aleksandar\Desktop\Json and Html\IDP4_plain_html\pool'
    ann_path = r'C:\Users\Aleksandar\Desktop\Json and Html\IDP4_members_json\pool\abojchevski'
    crf_path = r'F:\Projects\crfsuite'

    dataset = HTMLReader(html_path).read()

    NLTKSplitter().split(dataset)
    NLTKTokenizer().tokenize(dataset)

    ReadFromAnnJsonAnnotator(ann_path).annotate(dataset)
    TmVarRegexNLDefiner().define(dataset)
    print('/n'.join([ann.text for ann in dataset.annotations() if ann.is_nl])) #print the NL ones

    SimpleLabeler().label(dataset)
    SimpleFeatureGenerator().generate(dataset)

    crf = CRFSuite(crf_path)
    crf.create_input_file(dataset, 'train')
    crf.train()
    crf.create_input_file(dataset, 'test')
    crf.test()


