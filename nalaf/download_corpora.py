"""
Downloads the necessary NLTK corpora for nalaf.

Usage: ::

    $ python -m nalaf.download_corpora

"""
if __name__ == '__main__':
    import nltk

    CORPORA = ['punkt']

    for corpus in CORPORA:
        nltk.download(corpus)