from nala.features import FeatureGenerator
import re


class TmVarDefault(FeatureGenerator):
    """
    Generates simple CRF features based on the value of the token itself.
    For each token t, generates at most 5 features, corresponding to:
    * w[0] = the value of the words itself
    * w[1], w[2] = the values of the next two words in the sequence
    * w[-1], w[-2] = the values of the previous two words in the sequence

    Implements the abstract class FeatureGenerator.

    TODO: Instead of having a hard-coded template, allow the template to
    be optionally passed as a parameter.
    """

    def generate(self, dataset):
        """
        :type dataset: structures.data.Dataset
        """
        self.reg_spec_chars = re.compile(';:,.\->+_]')
        self.reg_chr_keys = re.compile('q|p|q[0-9]+|p[0-9]+|qter|pter|XY|t')
        self.reg_char_simple_bracket = re.compile('[\(\)]')
        self.reg_char_square_bracket = re.compile('[\[\]]')
        self.reg_char_curly_bracket = re.compile('[\{\}]')
        self.reg_char_slashes = re.compile('[\/\\\]')
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

            # nr of lettres
            token.features['num_alpha[0]'] = self.n_chars(token.word)

            # nr of specific chars: ";:,.->+_"
            token.features['num_spec_chars[0]'] = self.spec_chars(token.word)

            # chromosomal keytokens
            token.features['num_has_chr_key[0]'] = self.is_chr_key(token.word)

            #

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
        return self.reg_chr_keys.match(str)
