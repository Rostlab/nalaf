from nala.features import FeatureGenerator
import re


class TmVarFeatureGenerator(FeatureGenerator):
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

        # patterns
        self.reg_shape_uc = re.compile('[A-Z]')
        self.reg_shape_uc_plus = re.compile('[A-Z]+')

        self.reg_shape_lc = re.compile('[a-z]')
        self.reg_shape_lc_plus = re.compile('[a-z]+')

        self.reg_shape_nr = re.compile('[0-9]')
        self.reg_shape_nr_plus = re.compile('[0-9]+')

        self.reg_shape_chars = re.compile('[A-Za-z]')
        self.reg_shape_chars_plus = re.compile('[A-Za-z]+')
        # TODO re.search instead of re.match and exclude ".*" for regexs'

    def generate(self, dataset):
        """
        :type dataset: nala.structures.data.Dataset
        """
        last_token_str = ""
        for token in dataset.tokens():

            # nr of digits
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
            token.features['protein_symbols[0]'] = self.protein_symbols(token.word, last_token_str)

            # RScode
            token.features['rs_code[0]'] = self.rscode(token.word)

            # patterns
            # TODO patterns
            token.features['shape1[0]'] = self.rscode(token.word)

            # prefix patterns
            for index, value in enumerate(self.prefix_pattern(token.word)):
                token.features['prefix{}[0]'.format(index+1)] = value

            # suffix patterns
            for index, value in enumerate(self.suffix_pattern(token.word)):
                token.features['suffix{}[0]'.format(index+1)] = value

            # last token
            last_token_str = token.word

    def n_lower_chars(self, str):
        result = sum(1 for c in str if c.islower())
        return "L4+" if result > 4 else result

    def n_upper_chars(self, str):
        result = sum(1 for c in str if c.isupper())
        return "U4+" if result > 4 else result

    def n_nr_chars(self, str):
        result = sum(1 for c in str if c.isnumeric())
        return "N4+" if result > 4 else result

    def n_chars(self, str):
        result = sum(1 for c in str if c.isalpha())
        return "A4+" if result > 4 else result

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

    def protein_symbols(self, str, last_str):
        uc_tmp = str  # upper case
        lc_tmp = str.lower()  # lower case

        if self.reg_prot_symbols1.match(lc_tmp):
            return "ProteinSymFull"
        elif self.reg_prot_symbols2.match(lc_tmp):
            return "ProteinSymTri"
        elif self.reg_prot_symbols3.match(lc_tmp) and self.reg_prot_symbols4.match(last_str):
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

    def shape1(self, str):
        if not self.reg_spec_chars.match(str):
            pattern = self.reg_shape_uc.sub('A', str)
            pattern = self.reg_shape_lc.sub('a', pattern)
            pattern = self.reg_shape_nr.sub('0', pattern)
            return pattern
        return None

    def shape2(self, str):
        if not self.reg_spec_chars.match(str):
            pattern = self.reg_shape_chars.sub('a', str)
            pattern = self.reg_shape_nr.sub('0', pattern)
            return pattern
        return None

    def shape3(self, str):
        if not self.reg_spec_chars.match(str):
            pattern = self.reg_shape_uc_plus.sub('A', str)
            pattern = self.reg_shape_lc_plus.sub('a', pattern)
            pattern = self.reg_shape_nr_plus.sub('0', pattern)
            return pattern
        return None

    def shape4(self, str):
        if not self.reg_spec_chars.match(str):
            pattern = self.reg_shape_chars_plus.sub('a', str)
            pattern = self.reg_shape_nr_plus.sub('0', pattern)
            return pattern
        return None

    def prefix_pattern(self, str):
        prefix_array = []
        for x in range(1, 6):
            if len(str) >= x:
                prefix_array.append(str[:x])
            else:
                prefix_array.append(None)
        return prefix_array


    def suffix_pattern(self, str):
        suffix_array = []
        for x in range(1, 6):
            if len(str) >= x:
                suffix_array.append(str[-x:])
            else:
                suffix_array.append(None)
        return suffix_array

    # TODO as array or as string with spaces?
    # OPTIONAL discussion: should it be visible? P1:[pattern] or just [pattern] --> i would prefer visibility to actually be able to debug the results (but more data)


class TmVarDictionaryFeatureGenerator(FeatureGenerator):
    # TODO docsting
    """
    Implements the abstract class FeatureGenerator.
    """
    def __init__(self):
        self.patterns = [
            re.compile('([cgrm]\.[ATCGatcgu \/\>\<\?\(\)\[\]\;\:\*\_\-\+0-9]+(inv|del|ins|dup|tri|qua|con|delins|indel)[ATCGatcgu0-9\_\.\:]*)'),
            re.compile('(IVS[ATCGatcgu \/\>\<\?\(\)\[\]\;\:\*\_\-\+0-9]+(del|ins|dup|tri|qua|con|delins|indel)[ATCGatcgu0-9\_\.\:]*)'),
            re.compile('([cgrm]\.[ATCGatcgu \/\>\?\(\)\[\]\;\:\*\_\-\+0-9]+)'),
            re.compile('(IVS[ATCGatcgu \/\>\?\(\)\[\]\;\:\*\_\-\+0-9]+)'),
            re.compile('([cgrm]\.[ATCGatcgu][0-9]+[ATCGatcgu])'),
            re.compile('([ATCGatcgu][0-9]+[ATCGatcgu])'),
            re.compile('([0-9]+(del|ins|dup|tri|qua|con|delins|indel)[ATCGatcgu]*)'),
            re.compile('([p]\.[CISQMNPKDTFAGHLRWVEYX \/\>\<\?\(\)\[\]\;\:\*\_\-\+0-9]+(inv|del|ins|dup|tri|qua|con|delins|indel|fsX|fsx|fsx|fs)[CISQMNPKDTFAGHLRWVEYX \/\>\<\?\(\)\[\]\;\:\*\_\-\+0-9]*)'),
            re.compile('([p]\.[CISQMNPKDTFAGHLRWVEYX \/\>\?\(\)\[\]\;\:\*\_\-\+0-9]+)'),
            re.compile('([p]\.[A-Z][a-z]{0,2}[\W\-]{0,1}[0-9]+[\W\-]{0,1}[A-Z][a-z]{0,2})'),
            re.compile('([p]\.[A-Z][a-z]{0,2}[\W\-]{0,1}[0-9]+[\W\-]{0,1}(fs|fsx|fsX))')]

    def generate(self, dataset):
        """
        :type dataset: nala.structures.data.Dataset
        """
        for part in dataset.parts():
            so_far = 0
            matches = {}

            for index, pattern in enumerate(self.patterns):
                matches[index] = []
                for match in pattern.finditer(part.text):
                    matches[index].append((match.start(), match.end()))

            for sentence in part.sentences:
                for token in sentence:
                    so_far = part.text.find(token.word, so_far)

                    for match_index, match in matches.items():
                        token.features['pattern{}[0]'.format(match_index)] = 'O'
                        for start, end in match:
                            if start == so_far:
                                token.features['pattern{}[0]'.format(match_index)] = 'B'
                                break
                            elif start < so_far < end:
                                token.features['pattern{}[0]'.format(match_index)] = 'I'
                                break

