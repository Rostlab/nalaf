import csv
import re
from collections import Counter

def count_matches(dataset):
    with open('RegEx.NL') as file:
        reader = csv.reader(file, delimiter='\t')
        regexes = list(reader)

    matches = []

    for regex in regexes:
        if regex[0].startswith('(?-xism:'):
            try:
                regex[0] = re.compile(regex[0].replace('(?-xism:', ''),
                                      re.VERBOSE | re.IGNORECASE | re.DOTALL | re.MULTILINE)
            except:
                regex[0] = None
        else:
            regex[0] = re.compile(regex[0])


    for part in dataset.parts():
        for ann in part.annotations:
            for regex in regexes:
                if regex[0] is not None and regex[0].match(ann.text):
                    matches.append(regex[1])
                    break

    print(Counter(matches))