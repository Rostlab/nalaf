from nala.features import FeatureGenerator
import re


class TmVarDefault(FeatureGenerator):
    """
    Generates tmVar CRF features based on the value of the token itself.
    * w[0] = the value of the words itself
    This FeatureGenerator mainly uses Regular Expressions (which are initialized in the constructor).
    The generate function iterates over the dataset and creates the features which all get separately created in their own function.

    Hints:
    - when a feature is not found (as in: regex don't apply) then the feature gets the value None.

    Implements the abstract class FeatureGenerator.
    """
    def __init__(self):
        """
        Contains all regular expressions.
        """
        self.reg_spec_chars = re.compile('.*[-;:,.>+_].*')
        self.reg_chr_keys = re.compile('.*(q|p|q[0-9]+|p[0-9]+|qter|pter|XY|t).*')
        self.reg_char_simple_bracket = re.compile('.*[\(\)].*')
        self.reg_char_square_bracket = re.compile('.*[\[\]].*')
        self.reg_char_curly_bracket = re.compile('.*[\{\}].*')
        self.reg_char_slashes = re.compile('.*[\/\\\].*')
        self.reg_mutat_type = re.compile('.*(del|ins|dup|tri|qua|con|delins|indel).*')
        self.reg_frameshift_type = re.compile('.*(fs|fsX|fsx).*')
        self.reg_mutat_word = re.compile(
            '^(deletion|delta|elta|insertion|repeat|inversion|deletions|insertions|repeats|inversions).*')
        self.reg_mutat_article = re.compile(
            '^(single|a|one|two|three|four|five|six|seven|eight|nine|ten|[0-9]+|[0-9]+\.[0-9]+).*')
        self.reg_mutat_byte = re.compile('.*(kb|mb).*')
        self.reg_mutat_basepair = re.compile(
            '.*(base|bases|pair|amino|acid|acids|codon|postion|postions|bp|nucleotide|nucleotides).*')
        self.reg_type1 = re.compile('^[cgrm]$')
        self.reg_type12 = re.compile('^(ivs|ex|orf)$')
        self.reg_dna_symbols = re.compile('^[ATCGUatcgu]$')
        self.reg_prot_symbols1 = re.compile(
            '.*(glutamine|glutamic|leucine|valine|isoleucine|lysine|alanine|glycine|aspartate|methionine|threonine|histidine|aspartic|asparticacid|arginine|asparagine|tryptophan|proline|phenylalanine|cysteine|serine|glutamate|tyrosine|stop|frameshift).*')
        self.reg_prot_symbols2 = re.compile(
            '^(cys|ile|ser|gln|met|asn|pro|lys|asp|thr|phe|ala|gly|his|leu|arg|trp|val|glu|tyr|fs|fsx)$')
        self.reg_prot_symbols3 = re.compile('^(ys|le|er|ln|et|sn|ro|ys|sp|hr|he|la|ly|is|eu|rg|rp|al|lu|yr)$')
        self.reg_prot_symbols4 = re.compile('^[CISQMNPKDTFAGHLRWVEYX]$')
        self.reg_rs_code1 = re.compile('^(rs|RS|Rs)[0-9].*')
        self.reg_rs_code2 = re.compile('^(rs|RS|Rs)$')

    def generate(self, dataset):
        """
        :type dataset: structures.data.Dataset
        """
        # TODO last token
        for token in dataset.tokens():
            # nr of digits
            # TODO 0,1,2,3,4+ instead of len = nr
            token.features['num_nr[0]'] = self.n_nr_chars(token.word)

            # nr of upper case
            token.features['num_up[0]'] = self.n_upper_chars(token.word)

            # nr of lower case
            token.features['num_lo[0]'] = self.n_lower_chars(token.word)

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

            # type 1
            token.features['type1[0]'] = self.type1(token.word)

            # type 2
            token.features['type2[0]'] = self.type2(token.word)

            # dna symbols
            token.features['dna_symbols[0]'] = self.dna_symbols(token.word)

            # protein symbols
            token.features['protein_symbols[0]'] = self.protein_symbols(token.word)

            # RScode
            token.features['rs_code[0]'] = self.rscode(token.word)

            # patterns
            # TODO patterns

            # prefix patterns
            # TODO prefix patterns

            # suffix patterns
            # TODO suffix patterns

    def n_lower_chars(self, str):
        result = sum(1 for c in str if c.islower())
        return "L:4+" if result > 4 else result

    def n_upper_chars(self, str):
        result = sum(1 for c in str if c.isupper())
        return "U:4+" if result > 4 else result

    def n_nr_chars(self, str):
        result = sum(1 for c in str if c.isnumeric())
        return "N:4+" if result > 4 else result

    def n_chars(self, str):
        result = sum(1 for c in str if c.isalpha())
        return "A:4+" if result > 4 else result

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
        return "ChroKey" if self.reg_chr_keys.match(str) else None

    def mutation_type(self, str):
        lc_tmp = str.lower()

        # NOTE 2x if in code which makes no sense because first if gets always overwritten
        if self.reg_frameshift_type.match(lc_tmp):
            return "FrameShiftType"
        elif self.reg_mutat_type.match(lc_tmp):
            return "MutatType"
        else:
            return None

    def mutation_word(self, str):
        lc_tmp = str.lower()
        return "MutatWord" if self.reg_mutat_word.match(lc_tmp) else None

    def mutation_article_bp(self, str):
        mutat_article = ""  # NOTE is this programming ok?
        lc_tmp = str.lower()

        # NOTE was if -> base | if -> byte | elif -> bp | else -> None
        if self.reg_mutat_article.match(lc_tmp):
            mutat_article = "Base"
        elif self.reg_mutat_byte.match(lc_tmp):
            mutat_article = "Byte"
        elif self.reg_mutat_basepair.match(lc_tmp):
            mutat_article = "bp"
        else:
            mutat_article = None

        return mutat_article

    def type1(self, str):
        if self.reg_type1.match(str):
            return "Type1"
        elif self.reg_type12.match(str):
            return "Type1_2"
        else:
            return None

    def type2(self, str):
        return "Type2" if str == "p" else None

    def dna_symbols(self, str):
        return "DNASym" if self.reg_dna_symbols.match(str) else None

    def protein_symbols(self, str):
        uc_tmp = str  # upper case
        lc_tmp = str.lower()  # lower case

        if self.reg_prot_symbols1.match(lc_tmp):
            return "ProteinSymFull"
        elif self.reg_prot_symbols2.match(lc_tmp):
            return "ProteinSymTri"
        # TODO last token include: "&& $last_token[...]"
        elif self.reg_prot_symbols3.match(lc_tmp):
            return "ProteinSymTriSub"
        elif self.reg_prot_symbols4.match(uc_tmp):
            return "ProteinSymChar"
        else:
            return None

    def rscode(self, str):
        if self.reg_rs_code1.match(str):
            return "RSCode"
        elif self.reg_rs_code2.match(str):
            return "RSCode"
        else:
            return None
