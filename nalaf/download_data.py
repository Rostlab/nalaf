import nltk
from spacy.en import download as spacy_en_download

"""
Downloads the necessary data & corpora for nalaf.

Usage: ::

    $ python -m nalaf.download_data

"""
if __name__ == '__main__':

    CORPORA = ['punkt']

    for corpus in CORPORA:
        nltk.download(corpus)

    spacy_en_download.main(data_size='parser', force=False)
