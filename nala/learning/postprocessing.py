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
        regex_patterns = construct_regex_patterns_from_predictions(dataset)

        for doc_id, doc in dataset.documents.items():
            for part_id, part in doc.parts.items():
                self.__fix_issues(part)
                for regex in regex_patterns:
                    for match in regex.finditer(part.text):
                        start = match.start(1)
                        end = match.end(1)
                        matched_text = part.text[start:end]
                        ann = Annotation('e_2', start, matched_text)

                        Annotation.equality_operator = 'exact_or_overlapping'
                        if ann not in part.predicted_annotations:
                            if not self.short.search(matched_text) \
                                    and self.at_least_one_letter_n_number_letter_n_number.search(matched_text):
                                part.predicted_annotations.append(Annotation('e_2', start, matched_text))
                        elif ' ' not in matched_text:
                            Annotation.equality_operator = 'overlapping'
                            for index, ann_b in enumerate(part.predicted_annotations):
                                if ann == ann_b and len(matched_text) > len(ann_b.text):
                                    part.predicted_annotations[index] = ann

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
                if split[0]:
                    part.predicted_annotations.append(Annotation(ann.class_id, ann.offset, split[0]))
                if split[2]:
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
        item = item.replace('.', '\.').replace('+', '\+').replace(')', '\)').replace('(', '\(').replace('*', '\*')

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

        # append space before and after the constructed pattern
        regex_patterns.append(re.compile(' ({}) '.format(item)))

    # include already prepared regex patterns
    # modified by appending space before and after the original pattern
    with open(pkg_resources.resource_filename('nala.data', 'RegEx.NL')) as file:
        for regex in csv.reader(file, delimiter='\t'):
            if regex[0].startswith('(?-xism:'):
                try:
                    regex_patterns.append(re.compile(' ({}) '.format(regex[0].replace('(?-xism:', '')),
                                                     re.VERBOSE | re.IGNORECASE | re.DOTALL | re.MULTILINE))
                except:
                    pass
            else:
                regex_patterns.append(re.compile(' ({}) '.format(regex[0])))

    # add our own custom regex
    regex_patterns.append(re.compile(' ([ATCG][0-9]+[ATCG]/[ATCG]) '))

    return regex_patterns
