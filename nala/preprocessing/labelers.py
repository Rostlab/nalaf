import abc
from nala.structures.data import Label
import re
from nala.utils import MUT_CLASS_ID


class Labeler:
    """
    Abstract class for generating labels for each token in the dataset.
    Subclasses that inherit this class should:
    * Be named [Name]Labeler
    * Implement the abstract method label
    * Append new items to the list field "original_labels" of each Token in the dataset
    """

    @abc.abstractmethod
    def label(self, dataset):
        """
        :type dataset: nala.structures.data.Dataset
        """
        return


class BIOLabeler(Labeler):
    """
    Implements a simple labeler using the annotations of the dataset
    using the BIO (beginning, inside, outside) format. Creates labels
    based on the class_id value in the Annotation object. That is:
    * B-[class_id]
    * I-[class_id]
    * O

    Requires the list field "annotations" to be previously set.
    Implements the abstract class Labeler.
    """

    def label(self, dataset):
        """
        :type dataset: nala.structures.data.Dataset
        """
        for part in dataset.parts():
            for sentence in part.sentences:
                for token in sentence:
                    token.original_labels = [Label('O')]

                    for ann in part.annotations:
                        start = ann.offset
                        end = ann.offset + len(ann.text)
                        if start == token.start:
                            token.original_labels[0].value = 'B-{}'.format(ann.class_id)
                            break
                        elif start < token.start < end:
                            token.original_labels[0].value = 'I-{}'.format(ann.class_id)
                            break


class TmVarLabeler(Labeler):
    """
    Implements a labeler using the annotations of the dataset
    based on the labeling scheme used by tmVar (http://www.ncbi.nlm.nih.gov/CBBresearch/Lu/Demo/tmTools/#tmVar)

    This is however a close approximation of their lableing scheme and not an exact replica.

    The following labels are possible:
    * A - Reference sequence
    * T - Mutation type
    * F - Frame shift
    * R - SNP
    * M - Mutant
    * W - Wild type
    * S - Frame shift position
    * P - Mutation position
    * I - Other inside mutation tokens
    * O - Outside

    Requires the list field "annotations" to be previously set.
    Requires the TmVarTokenizer to be used.
    Implements the abstract class Labeler.
    """

    def __init__(self):
        # A
        self.label_reference_sequence = re.compile('(^[cgrmp]$)|(^(ivs|ex|orf)$)')
        # T
        self.label_type = re.compile('(del|ins|dup|tri|qua|con|delins|indel)')
        # F
        self.label_frameshift = re.compile('^(fs|fsX|fsx)$')
        # R
        self.label_snip = re.compile('^(rs|RS|Rs)$')

        # W or M (wild type or mutant)
        self.dna_symbols = re.compile('^[ATCGUatcgu]+$')
        self.protein_symbols = re.compile('(glutamine|glutamic|leucine|valine|isoleucine|lysine|alanine|glycine|'
                                          'aspartate|methionine|threonine|histidine|aspartic|asparticacid|arginine|'
                                          'asparagine|tryptophan|proline|phenylalanine|cysteine|serine|glutamate|'
                                          'tyrosine|stop|frameshift)|(^(cys|ile|ser|gln|met|asn|pro|lys|asp|thr|phe|'
                                          'ala|gly|his|leu|arg|trp|val|glu|tyr|fs|fsx)$)|(^[cisqmnpkdtfaghlrwveyx]$)')

        # P or S (mutation_position or frameshift_position)
        self.position = re.compile('^[0-9]+$')

    def _match_regex_label(self, previous_token, token):
        if self.label_reference_sequence.search(token.word):
            token.original_labels[0].value = 'A'  # Reference sequence
        elif self.label_type.search(token.word):
            token.original_labels[0].value = 'T'  # Mutation type
        elif self.label_frameshift.search(token.word):
            token.original_labels[0].value = 'F'  # Frame shift
        elif previous_token is not None and previous_token.original_labels[0].value is 'F' and token.word is 'X':
            token.original_labels[0].value = 'F'  # Frame shift
        elif self.label_snip.search(token.word):
            token.original_labels[0].value = 'R'  # SNP
        elif self.dna_symbols.search(token.word) or self.protein_symbols.search(token.word.lower()):
           token.original_labels[0].value = '*'   # temporary for W or M (wild type or mutant)
        elif self.position.search(token.word):
            if previous_token is not None and previous_token.original_labels[0].value == 'F':
                token.original_labels[0].value = 'S'  # Frame shift position
            else:
                token.original_labels[0].value = 'P'  # Mutation position
        else:
            token.original_labels[0].value = 'I'  # Other inside mutation tokens

    def label(self, dataset):
        """
        :type dataset: nala.structures.data.Dataset
        """
        for part in dataset.parts():
            previous_token = None
            for sentence in part.sentences:
                alternate = 'W'
                for token in sentence:
                    token.original_labels = [Label('O')]

                    for ann in part.annotations:
                        start = ann.offset
                        end = ann.offset + len(ann.text)
                        if start == token.start or start < token.start < end:
                            if ann.class_id == MUT_CLASS_ID:
                                self._match_regex_label(previous_token, token)
                                previous_token = token

                                # replace temporary label with W or M
                                if token.original_labels[0].value == '*':
                                    token.original_labels[0].value = alternate
                                    alternate = 'W' if alternate == 'M' else 'M'
                                # reset the alternation to W since we reached end
                                if token.end == end:
                                    alternate = 'W'
                                break

                # iterate a sliding window of 3
                # when you find 'P I P' labels replace them with 'P P P'
                for previous, current, next in zip(sentence, sentence[1:], sentence[2:]):
                    if previous.original_labels[0].value == 'P' and next.original_labels[0].value == 'P':
                        if current.original_labels[0].value == 'I':
                            current.original_labels[0].value = 'P'


class BIEOLabeler(Labeler):
    """
    Implements a simple labeler using the annotations of the dataset
    using the BIEO (beginning, inside, ending, outside) format. Creates labels
    based on the class_id value in the Annotation object. That is:
    * B-[class_id]
    * I-[class_id]
    * E-[class_id]
    * O

    Requires the list field "annotations" to be previously set.
    Implements the abstract class Labeler.
    """

    def label(self, dataset):
        """
        :type dataset: nala.structures.data.Dataset
        """
        for part in dataset.parts():
            for sentence in part.sentences:
                for token in sentence:
                    token.original_labels = [Label('O')]

                    for ann in part.annotations:
                        start = ann.offset
                        end = ann.offset + len(ann.text)
                        if start == token.start:
                            token.original_labels[0].value = 'B-{}'.format(ann.class_id)
                            break
                        elif start < token.start < token.end < end:
                            token.original_labels[0].value = 'I-{}'.format(ann.class_id)
                            break
                        elif token.end == end:
                            token.original_labels[0].value = 'E-{}'.format(ann.class_id)
                            break
