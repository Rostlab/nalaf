from nala.learning.evaluators import find_offsets
import pkg_resources
import csv


def predict_with_regex_patterns(dataset):
    regex_patterns = construct_regex_patterns_from_predictions(dataset)

    predicted_offsets = []

    for part_id, part in dataset.partids_with_parts():
        for regex in regex_patterns:
            for match in regex.finditer(part.text):
                predicted_offsets.append((match.start(), match.end(), part_id))

    return predicted_offsets


def construct_regex_patterns_from_predictions(dataset):
    import re
    _, _, predicted_items = find_offsets(dataset)

    regex_patterns = []
    for item in predicted_items:
        # numbers pattern
        item = re.sub('[0-9]+', '[0-9]+', item)

        # take care of special tokens
        item = re.sub('(IVS|EX)', '@@@@', item)
        item = re.sub('(rs|ss)', '@@@', item)

        # letters pattern
        item = re.sub('[a-z]', '[a-z]', item)
        item = re.sub('[A-Z]', '[A-Z]', item)

        # revert back special tokens
        item = re.sub('@@@@', '(IVS|EX)', item)
        item = re.sub('@@@', '(rs|ss)', item)

        # escape special regex characters
        item = item.replace('.', '\.').replace('+', '\+').replace(')', '\)').replace('(', '\(')
        regex_patterns.append(re.compile(item))

    # include already prepared regex patterns
    with open(pkg_resources.resource_filename('nala.data', 'RegEx.NL')) as file:
        for regex in csv.reader(file, delimiter='\t'):
            if regex[0].startswith('(?-xism:'):
                try:
                    regex_patterns.append(re.compile(regex[0].replace('(?-xism:', ''),
                                                       re.VERBOSE | re.IGNORECASE | re.DOTALL | re.MULTILINE))
                except:
                    pass
            else:
                regex_patterns.append(re.compile(regex[0]))

    return regex_patterns
