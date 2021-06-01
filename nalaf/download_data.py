import nltk
from spacy.en import download as spacy_en_download
import sys

"""
Downloads the necessary data & corpora for nalaf.

Usage: ::

    $ python -m nalaf.download_data

"""
if __name__ == '__main__':

    # If you get a an error like "NLTK download SSL: Certificate verify failed" or "ssl.SSLError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed (_ssl.c:749)"
    # ... take a look at this solution: https://stackoverflow.com/a/41692664/341320

    # Download NLTK models

    NLTK_CORPORA = ['punkt']

    for corpus in NLTK_CORPORA:
        nltk.download(corpus)

    # Download Spacy models
    try:
        spacy_en_download.main(data_size='parser', force=False)
    except Exception:
        import traceback
        traceback.print_exc()

        import os
        import spacy

        model = "en-1.1.0"
        # python -c "import os; import spacy; print(os.path.join(os.path.dirname(spacy.__file__), 'data'))"
        spacy_data_path = os.path.join(os.path.dirname(spacy.__file__), 'data')

        print("ERROR Could not save spacy English parser model. Download model: {} --> and extract it in: {}".format(model, spacy_data_path), file=sys.stderr)

        # TODO (2021-06-01 JMC) download & fix error: ERROR Could not save spacy English parser model. Download model: en-1.1.0 --> and extract it in: /Users/juanmirocks/Library/Caches/pypoetry/virtualenvs/nalaf-dHMIkhB4-py3.6/lib/python3.6/site-packages/spacy/data

    # TODO download non-packaged [biolemmatizer-core-1.2-jar-with-dependencies.jar](https://github.com/Rostlab/nalaf/blob/develop/nalaf/data/biolemmatizer-core-1.2-jar-with-dependencies.jar)

    # TODO download non-packaged [example_entity_model](https://github.com/Rostlab/nalaf/blob/develop/nalaf/data/example_entity_model)
