import argparse
import configparser
import functools
import sys
import json
import re
import timeit
import time

from nala.utils.readers import HTMLReader, SETHReader, TmVarReader, VerspoorReader
from nala.preprocessing.spliters import NLTKSplitter
from nala.preprocessing.tokenizers import NLTKTokenizer
from nala.utils.annotation_readers import AnnJsonAnnotationReader, SETHAnnotationReader
from nala.preprocessing.labelers import BIOLabeler, BIEOLabeler, TmVarLabeler
from nala.preprocessing.definers import TmVarRegexNLDefiner
from nala.preprocessing.definers import ExclusiveNLDefiner, SimpleExclusiveNLDefiner
from nala.preprocessing.definers import TmVarNLDefiner
from nala.preprocessing.definers import InclusiveNLDefiner
from nala.utils.writers import StatsWriter
from nala.features.simple import SimpleFeatureGenerator
from nala.features.tmvar import TmVarFeatureGenerator
from nala.features.window import WindowFeatureGenerator
from nala.learning.crfsuite import CRFSuite

import nala.utils.db_validation as dbcheck
from nala.utils.writers import TagTogFormat
from nala.utils.dataset_selection import RegexSelector, TmVarRegexCombinedSelector
from nala.utils.writers import PubTatorFormat

html_path = "resources/corpora/idp4/html"
ann_path = "resources/corpora/idp4/annjson"
crf_path = "/usr/local/Cellar/crfsuite/0.12"

dataset = HTMLReader(html_path).read()

AnnJsonAnnotationReader(ann_path).annotate(dataset)

# NLTKSplitter().split(dataset)
# NLTKTokenizer().tokenize(dataset)

ExclusiveNLDefiner().define(dataset)

# PubTatorFormat(dataset, no_annotations=False).export()

print(dataset)

nl_annotations = []

# import connecting_words.json
with open('nala/data/connecting_words.json', 'r') as f:
    regexs = json.load(f)

# print(regexs)
compiled_regexs = [re.compile(x) for x in regexs]

nr_word_regex = re.compile('\\b(one|two|three|four|five|six|seven|eight|nine|ten)\\b')
aa_short_regex = re.compile('\\b(cys|ile|ser|gln|met|asn|pro|lys|asp|thr|phe|ala|gly|his|leu|arg|trp|val|glu|tyr)\\b')
aa_long_regex = re.compile(
    '\\b(glutamine|glutamic acid|leucine|valine|isoleucine|lysine|alanine|glycine|aspartate|methionine|threonine|histidine|aspartic acid|arginine|asparagine|tryptophan|proline|phenylalanine|cysteine|serine|glutamate|tyrosine)\\b')
bp_code = re.compile('\\b\\w\\b')

wordlist = []

for ann in dataset.annotations():
    if ann.subclass == 1 or ann.subclass == 2:
        new_text = ann.text.lower()
        for reg in compiled_regexs:
            new_text = reg.sub('_TT_', new_text)
        # re.sub('\\b\\d+\\b]', '_NR_', new_text)
        new_text = re.sub('\\b\\w*\\d+\\w*\\b', '_CODE_', new_text)
        new_text = nr_word_regex.sub('_TT_', new_text)
        new_text = aa_short_regex.sub('_AA_', new_text)
        new_text = aa_long_regex.sub('_AA_', new_text)
        new_text = bp_code.sub('_TT_', new_text)
        # new_text = re.sub('\\W', ' ', new_text)
        # new_text = re.sub('\\b(\\w{1,3})\\b', '_TT_', new_text)

        wordlist.extend(new_text.split(' '))
        # print(new_text)
        nl_annotations.append(new_text)

wordset = set(wordlist)
wordlist = sorted(list(wordset))
# print(json.dumps(wordlist, indent=2, sort_keys=True))
# print(json.dumps(nl_annotations, indent=2, sort_keys=True))

# read in nl_patterns
with open('nala/data/nl_patterns.json', 'r') as f:
    regexs = json.load(f)

patterns = [re.compile(x) for x in regexs]

# check for annotations

# for part in dataset.parts():
#     print(part.text)

# dataset with tmVar
dataset_high_recall = TmVarReader('resources/corpora/idp4/pubtator_tmvar.txt').read()
TP = 0
FP = 0
for did, doc in dataset.documents.items():
    part_offset = 0
    for i, x in enumerate(doc.parts):
        # print("Part", i)
        sent_offset = 0
        cur_part = doc.parts.get(x)
        new_text = cur_part.text.lower()
        new_text = re.sub('\s+', ' ', new_text)
        sentences = new_text.split('. ')
        for sent in sentences:
            part_length = len(sent)
            new_text = re.sub('\W+', ' ', sent)
            for i, reg in enumerate(patterns):
                # print(reg.pattern)
                lasttime = time.time()
                match = reg.search(new_text)
                if match:
                    if did in dataset_high_recall.documents:
                        other_doc = dataset_high_recall.documents.get(did)
                        if not other_doc.overlaps_with_mention(match.span()[0]) and not other_doc.overlaps_with_mention(
                                match.span()[1]):
                            print(sent, match, reg.pattern)
                            if doc.overlaps_with_mention(match.span()[0]) and doc.overlaps_with_mention(
                                    match.span()[1]):
                                TP += 1
                            else:
                                FP += 1
                            break
                if lasttime - time.time() > 1:
                    print(i)
            sent_offset += 2 + part_length
        part_offset += sent_offset
    print('TP:{0}, FP:{1}, %containingNLmentions:{2}'.format(TP, FP, TP/(TP + FP)))
