import re

from nalaf.features import FeatureGenerator


class RegexNLFeatureGenerator(FeatureGenerator):
    def __init__(self):
        self.patterns = [
            re.compile('[g]\.[0-9]+_[0-9]+(del)[0-9]+'),
            re.compile('deletion of( (the|a))?.* region'),
            re.compile('deletion of( (the|a))?( \d+(bp|base pairs?|a\.a\.|amino acids?|nucleotides?)?)? [\w\-\.]+'),
            re.compile('\d+(-| )?(bp|base pairs?|a\.a\.|amino acids?|nucleotides?).*deletion'),
            re.compile('[\w\-\.]+ deletion'),
            re.compile('(c|carboxyl?|cooh|n|amino|nh2|amine)(-| )(terminus|terminal)( (tail|end))?'),
            re.compile('exons? \d+(( ?(and|or|-) ?\d+))?')
        ]

    def generate(self, dataset):
        """
        :type dataset: nala.structures.data.Dataset
        """
        for part in dataset.parts():
            matches = {}

            for index, pattern in enumerate(self.patterns):
                matches[index] = []
                for match in pattern.finditer(part.text):
                    matches[index].append((match.start(), match.end()))

            for sentence in part.sentences:
                for token in sentence:
                    for match_index, match in matches.items():
                        name = 'regex_nl_{}'.format(match_index)
                        value = 'O'
                        for start, end in match:
                            if start == token.start:
                                value = 'B'
                                break
                            elif start < token.start < token.end < end:
                                value = 'I'
                                break
                            elif token.end == end:
                                value = 'E'
                                break

                        token.features[name] = value
