import csv
import re
from collections import Counter

def count_matches(dataset):
    with open('RegEx.NL') as file:
        reader = csv.reader(file, delimiter='\t')
        regexes = list(reader)

    matches = []

    for part in dataset.parts():
        for ann in part.annotations:
            for regex in regexes:
                try:
                    if re.match(regex[0],ann.text):
                        matches.append(regex[1])
                        break
                except:
                    pass

    print(Counter(matches))