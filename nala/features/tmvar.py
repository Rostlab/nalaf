from nala.features import FeatureGenerator
import re


class TmVarDefault(FeatureGenerator):
    """
    Generates tmVar CRF features based on the value of the token itself.
    * w[0] = the value of the words itself

    Implements the abstract class FeatureGenerator.
    """

    def generate(self, dataset):
        """
        :type dataset: structures.data.Dataset
        """
        self.reg_spec_chars = re.compile('.*[-;:,.>+_].*')
        self.reg_chr_keys = re.compile('.*(q|p|q[0-9]+|p[0-9]+|qter|pter|XY|t).*')
        self.reg_char_simple_bracket = re.compile('.*[\(\)].*')
        self.reg_char_square_bracket = re.compile('.*[\[\]].*')
        self.reg_char_curly_bracket = re.compile('.*[\{\}].*')
        self.reg_char_slashes = re.compile('.*[\/\\\].*')
        self.reg_mutat_type = re.compile('.*(del|ins|dup|tri|qua|con|delins|indel).*')
        self.reg_frameshift_type = re.compile('.*(fs|fsX|fsx).*')
        self.reg_mutat_word = re.compile('^(deletion|delta|elta|insertion|repeat|inversion|deletions|insertions|repeats|inversions).*')
        self.reg_mutat_article = re.compile('^(single|a|one|two|three|four|five|six|seven|eight|nine|ten|[0-9]+|[0-9]+\.[0-9]+).*')
        self.reg_mutat_byte = re.compile('.*(kb|mb).*')
        self.reg_mutat_basepair = re.compile('.*(base|bases|pair|amino|acid|acids|codon|postion|postions|bp|nucleotide|nucleotides).*')

        for token in dataset.tokens():
            # nr of digits
            # TODO 0,1,2,3,4+ instead of len = nr
            token.features['num_nr[0]'] = self.n_nr_chars(token.word)

            # nr of upper case
            token.features['num_up[0]'] = self.n_upper_chars(token.word)

            # nr of lower case
            token.features['num_lo[0]'] = self.n_lower_chars(token.word)

            # nr of chars
            token.features['length[0]'] = len(token.word)

            # nr of letters
            token.features['num_alpha[0]'] = self.n_chars(token.word)

            # nr of specific chars: ";:,.->+_"
            token.features['num_spec_chars[0]'] = self.spec_chars(token.word)

            # chromosomal keytokens
            token.features['num_has_chr_key[0]'] = self.is_chr_key(token.word)

            # mutation type
            token.features['mutat_type[0]'] = self.mutation_type(token.word)

            # mutation word
            token.features['mutat_word[0]'] = self.mutation_word(token.word)

            # mutation article and basepair
            token.features['mutat_article_bp[0]'] = self.mutation_article_bp(token.word)

    # TODO check if ok the implementation (edge cases e.g. numeric means 123.232? or 123 and 232?)
    def n_lower_chars(self, str):
        return sum(1 for c in str if c.islower())

    def n_upper_chars(self, str):
        return sum(1 for c in str if c.isupper())

    def n_nr_chars(self, str):
        return sum(1 for c in str if c.isnumeric())

    def n_chars(self, str):
        return sum(1 for c in str if c.isalpha())

    def spec_chars(self, str):
        if self.reg_spec_chars.match(str):
            return "SpecC1"
        elif self.reg_char_simple_bracket.match(str):
            return "SpecC2"
        elif self.reg_char_curly_bracket.match(str):
            return "SpecC3"
        elif self.reg_char_square_bracket.match(str):
            return "SpecC4"
        elif self.reg_char_slashes.match(str):
            return "SpecC5"
        else:
            return None

    def is_chr_key(self, str):
        return True if self.reg_chr_keys.match(str) else None

    def mutation_type(self, str):
        if self.reg_frameshift_type.match(str):  # NOTE check if this is as in code (2x if but not if else)
            return "FrameShiftType"
        elif self.reg_mutat_type.match(str):
            return "MutatType"
        else:
            return None

    def mutation_word(self, str):
        return "MutatWord" if self.reg_mutat_word.match(str) else None

    def mutation_article_bp(self, str):
        mutat_article = ""  # NOTE is this programming ok?

        if self.reg_mutat_article.match(str):
            mutat_article = "Base"
        if self.reg_mutat_byte.match(str):
            mutat_article = "Byte"
        elif self.reg_mutat_basepair.match(str):
            mutat_article = "bp"
        else:
            mutat_article = None

        return mutat_article