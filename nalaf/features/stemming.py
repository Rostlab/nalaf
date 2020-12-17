from nalaf.features import FeatureGenerator
from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from subprocess import Popen, PIPE
import os
import fcntl
import os.path
import pkg_resources
from nalaf import print_debug
from nalaf.features import get_spacy_nlp_english


class SpacyLemmatizer(FeatureGenerator):
    """
    Lemmatize using spacy default English lemmatizer

    Note: the lemma is stored as feature 'stem' for consistency with other parts.
    """

    def __init__(self):
        self.nlp = get_spacy_nlp_english()

    def generate(self, dataset):
        """
        :type dataset: nalaf.structures.data.Dataset
        """
        for part in dataset.parts():
            for sentence in part.sentences:
                text_tokens = list(map(lambda x: x.word, sentence))
                spacy_doc = self.nlp.tokenizer.tokens_from_list(text_tokens)

                self.nlp.tagger(spacy_doc)  # this we need, otherwise the lemma is empty

                for token, spacy_token in zip(sentence, spacy_doc):
                    token.features['stem'] = spacy_token.lemma_  # already in lower case


class BioLemmatizer(FeatureGenerator):
    """
    Uses BioLemmatizer (http://biolemmatizer.sourceforge.net)
    to set the feature 'stem' (actually a lemma) for every token

    NOTE: requires Java 6 installed on the system.

    NOTE: the needed biolemmatizer jar is included in the src nalaf distribution, but not packaged into PyPi
    """

    @staticmethod
    def __setNonBlocking(fd):
        """
        Set the file description of the given file descriptor to non-blocking.
        """
        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        flags = flags | os.O_NONBLOCK
        fcntl.fcntl(fd, fcntl.F_SETFL, flags)

    def __init__(self):
        self.jar_path = pkg_resources.resource_filename('nalaf.data',
                                                        "biolemmatizer-core-1.2-jar-with-dependencies.jar")
        if not os.path.isfile(self.jar_path):
            raise Exception("Could't find biolemmatizer jar: " + self.jar_path)

        self.program = ["java", "-Xmx1G", "-jar", self.jar_path, "-l", "-t"]
        self.p = Popen(self.program, universal_newlines=True, stdin=PIPE, stdout=PIPE, stderr=PIPE, bufsize=1)
        BioLemmatizer.__setNonBlocking(self.p.stdout)
        BioLemmatizer.__setNonBlocking(self.p.stderr)

        # Initialize java program
        print_debug("BioLemmatizer: INIT START")
        out = None
        while not out:
            try:
                out = self.p.stdout.read()
            except TypeError:
                continue
            else:
                if ("Running BioLemmatizer in interactive mode" in out):
                    break
                else:
                    out = None

        print_debug("BioLemmatizer: INIT END")

    def generate_word(self, word, postag):
        self.p.stdin.write(word + " " + postag + "\n")
        out = None
        while not out:
            try:
                out = self.p.stdout.read()
                out = out.strip().lower()
            except TypeError:
                continue
            else:
                if out:
                    return out

    def generate(self, dataset):
        """
        :type dataset: nalaf.structures.data.Dataset
        """
        for token in dataset.tokens():
            token.features['stem'] = self.generate_word(token.word, token.features['tag[0]'])


PORTER_STEMMER = PorterStemmer()
ENGLISH_STEMMER = SnowballStemmer("english")


class PorterStemFeatureGenerator(FeatureGenerator):
    """
    Generates stem features based on the values of the tokens themselves.
        * stem[0] = the value of the word itself stemmed

    Uses the NLTK implementation of the Porter Stemmer. The original Porter Stemming
    algorithm can be found here http://tartarus.org/~martin/PorterStemmer/.

    Implements the abstract class FeatureGenerator.
    """

    def __init__(self):
        self.stemmer = PORTER_STEMMER

    def generate(self, dataset):
        """
        :type dataset: nalaf.structures.data.Dataset
        """
        for token in dataset.tokens():
            token.features['stem'] = self.stemmer.stem(token.word.lower())
