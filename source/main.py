from utils.readers import HTMLReader
from preprocessing.spliters import NTLKSplitter
from preprocessing.tokenizers import NLTKTokenizer
from preprocessing.annotators import ReadFromAnnJsonAnnotator
from preprocessing.labelers import SimpleLabeler
from features.simple import SimpleFeatureGenerator
from utils.crfsuite import CRFSuite

if __name__ == "__main__":

    html_path = r'C:\Users\Aleksandar\Desktop\Json and Html\IDP4_plain_html\pool'
    ann_path = r'C:\Users\Aleksandar\Desktop\Json and Html\IDP4_members_json\pool\abojchevski'
    crf_path = r'F:\Projects\crfsuite'

    dataset = HTMLReader(html_path).read()

    NTLKSplitter().split(dataset)
    NLTKTokenizer().tokenize(dataset)

    ReadFromAnnJsonAnnotator(ann_path).annotate(dataset)
    SimpleLabeler().label(dataset)
    SimpleFeatureGenerator().generate(dataset)

    crf = CRFSuite(crf_path)
    crf.create_input_file(dataset, 'train')
    crf.train()
    crf.create_input_file(dataset, 'test')
    crf.test()


