"""
Downloads the necessary NLTK corpora for nalaf.

Usage: ::

    $ python -m nalaf.download_corpora

"""
if __name__ == '__main__':
    from nltk import download

    CORPORA = ['punkt']

    for corpus in CORPORA:
        download(corpus)
