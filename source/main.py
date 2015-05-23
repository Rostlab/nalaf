from utils.readers import HTMLReader
from preprocessing.spliters import NTLKSplitter
from preprocessing.tokenizers import NLTKTokenizer
from preprocessing.annotators import ReadFromAnnJsonAnnotator

if __name__ == "__main__":

    path = r'C:\Users\Aleksandar\Desktop\Json and Html\IDP4_plain_html\pool'
    ann_path = r'C:\Users\Aleksandar\Desktop\Json and Html\IDP4_members_json\pool\abojchevski'

    dataset = HTMLReader(path).read()
    NTLKSplitter().split(dataset)
    NLTKTokenizer().tokenize(dataset)
    ReadFromAnnJsonAnnotator(ann_path).annotate(dataset)

    for sentence in dataset.sentences():
        print(sentence)



