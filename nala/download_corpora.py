"""
Downloads the necessary NLTK corpora for nala.

Usage: ::

    $ python -m textblob.download_corpora

"""
if __name__ == '__main__':
    import nltk

    CORPORA = ['punkt']

    for corpus in CORPORA:
        nltk.download(corpus)