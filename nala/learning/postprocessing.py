import pkg_resources
import csv
import re
from nala.learning.evaluators import is_overlapping
from nala.structures.data import Annotation


class PostProcessing:
    def __init__(self):
        self.at_least_one_letter_n_number_letter_n_number = re.compile('(?=.*[A-Za-z])(?=.*[0-9])[A-Za-z0-9]+')
        self.short = re.compile('^[A-Z][0-9][A-Z]$')

    def process(self, dataset):
        existing_predictions = []

        for doc_id, doc in dataset.documents.items():
            for part_id, part in doc.parts.items():
                self.__fix_issues(part)

                for index, ann in enumerate(part.predicted_annotations):
                    existing_predictions.append((ann.offset, ann.offset + len(ann.text), part_id, ann.class_id, doc_id, index))

        regex_patterns = construct_regex_patterns_from_predictions(dataset)

        for doc_id, doc in dataset.documents.items():
            for part_id, part in doc.parts.items():
                for regex in regex_patterns:
                    for match in regex.finditer(part.text):
                        offset = (match.start(), match.end(), part_id, 'e_2', doc_id)
                        matched_text = part.text[match.start():match.end()]

                        # TODO Refactor into regex instead of check
                        try:
                            space_before = part.text[match.start() - 1] == ' '
                        except IndexError:
                            space_before = True
                        try:
                            space_after = part.text[match.end()] == ' '
                        except IndexError:
                            space_after = True

                        if not is_overlapping(offset, existing_predictions):
                            if not self.short.search(matched_text) and space_before and space_after \
                                    and self.at_least_one_letter_n_number_letter_n_number.search(matched_text):
                                existing_predictions.append(offset)
                                part.predicted_annotations.append(Annotation('e_2', match.start(), matched_text))
                        else:
                            # TODO Refactor to return an object
                            # our custom part (needs to be optimized) adds 1% to the f_measure (eg. 87 to 88)
                            for offset_b in existing_predictions:
                                # if there is a partial overlap with a regex match
                                if offset[2:5] == offset_b[2:5] and offset[0] <= offset_b[1] and offset[1] >= offset_b[0]:
                                    # but that overlap does not contain spaces and cover text span bigger in length
                                    if (offset[1] - offset[0]) > (offset_b[1]-offset_b[0]) and ' ' not in matched_text:
                                        # and additionally it's between two spaces
                                        if re.search(' +{} +'.format(regex.pattern), part.text):
                                            # replace the existing one by the one found one since it is probably better
                                            dataset.documents[offset_b[-2]].parts[offset_b[-4]].predicted_annotations[offset_b[-1]] \
                                                = Annotation('e_2', match.start(), matched_text)

    def __fix_issues(self, part):
        to_be_removed = []
        for index, ann in enumerate(part.predicted_annotations):
            start = ann.offset
            end = ann.offset + len(ann.text)

            # split multiple mentions
            if re.search(' *(/) *', ann.text):
                split = re.split(' *(/) *', ann.text)

                if self.at_least_one_letter_n_number_letter_n_number.search(split[0]) \
                        and self.at_least_one_letter_n_number_letter_n_number.search(split[2]):
                    to_be_removed.append(index)
                    part.predicted_annotations.append(Annotation(ann.class_id, ann.offset, split[0]))
                    part.predicted_annotations.append(
                        Annotation(ann.class_id, part.text.find(split[2], ann.offset), split[2]))

            # split multiple mentions
            if re.search(' *(and|,|or) *', ann.text):
                to_be_removed.append(index)
                split = re.split(' *(and|,|or) *', ann.text)
                part.predicted_annotations.append(Annotation(ann.class_id, ann.offset, split[0]))
                part.predicted_annotations.append(
                    Annotation(ann.class_id, part.text.find(split[2], ann.offset), split[2]))

            # fix boundary #17000021	251	258	1858C>T --> +1858C>T
            if re.search('^[0-9]', ann.text) and re.search('([\-\+])', part.text[start - 1]):
                ann.offset -= 1
                ann.text = part.text[start - 1] + ann.text

            # fix boundary delete (
            if ann.text[0] == '(' and ')' not in ann.text:
                ann.offset += 1
                ann.text = ann.text[1:]

            # fix boundary delete )
            if ann.text[-1] == ')' and '(' not in ann.text:
                ann.text = ann.text[:-1]

            # fix boundary add missing (
            if part.text[start - 1] == '(' and ')' in ann.text:
                ann.offset -= 1
                ann.text = '(' + ann.text

            # fix boundary add missing )
            try:
                if part.text[end] == ')' and '(' in ann.text:
                    ann.text += ')'
            except IndexError:
                pass
        part.predicted_annotations = [ann for index, ann in enumerate(part.predicted_annotations)
                                      if index not in to_be_removed]

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

    # add our own custom regex
    regex_patterns.append(re.compile('[ATCG][0-9]+[ATCG]/[ATCG]'))

    return regex_patterns
