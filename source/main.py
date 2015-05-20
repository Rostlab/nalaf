from utils.readers import HTMLReader
from preprocessing.spliters import NTLKSplitter
from preprocessing.tokenizers import NLTKTokenizer


if __name__ == "__main__":

    dataset = HTMLReader(r'C:\Users\Aleksandar\Desktop\Json and Html\IDP4_plain_html\pool').read()
    NTLKSplitter().split(dataset)
    NLTKTokenizer().tokenize(dataset)

    for doc in dataset:
        for part in doc:
            print(part.sentences)
