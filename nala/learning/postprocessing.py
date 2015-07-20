import pkg_resources
import csv
import re
from nala.learning.evaluators import _is_overlapping
from nala.structures.data import Annotation

def predict_with_regex_patterns(dataset):
    # TODO figure out how to best set class_id independent since we don't know it
    """
    :type dataset: nala.structures.data.Dataset
    """
    regex_patterns = construct_regex_patterns_from_predictions(dataset)

    existing_predictions = []
    for part_id, part in dataset.partids_with_parts():
        for ann in part.predicted_annotations:
            existing_predictions.append((ann.offset, ann.offset + len(ann.text), part_id, ann.class_id))

    filter_short = re.compile('^[A-Z][0-9][A-Z]')

    for part_id, part in dataset.partids_with_parts():
        for regex in regex_patterns:
            for match in regex.finditer(part.text):
                offset = (match.start(), match.end(), part_id, 'e_2')
                matched_text = part.text[match.start():match.end()]

                if not _is_overlapping(offset, existing_predictions) and not filter_short.match(matched_text):
                    existing_predictions.append(offset)
                    part.predicted_annotations.append(
                        Annotation('e_2', match.start(), matched_text))


def construct_regex_patterns_from_predictions(dataset):
    """
    :type dataset: nala.structures.data.Dataset
    """

    regex_patterns = []
    for ann in dataset.predicted_annotations():
        item = ann.text
        # escape special regex characters
        item = item.replace('.', '\.').replace('+', '\+').replace(')', '\)').replace('(', '\(')

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
