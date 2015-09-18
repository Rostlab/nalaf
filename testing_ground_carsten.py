import argparse
import configparser
import functools
import sys
import json
import re
import timeit
import time
from nala import print_verbose

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
        new_text = re.sub('\\W', ' ', new_text)
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
_length = len(dataset.documents.values())
_progress = 0
_timestart = time.time()

_time_avg_per_pattern = 0
_pattern_calls = 0
_time_reg_pattern_total = 0
_time_max_pattern = 0
_low_performant_pattern = ""

with open('results/testing_ground_carsten.txt', 'w', encoding='utf-8') as f:
    for did, doc in dataset.documents.items():
        part_offset = 0
        for i, x in enumerate(doc.parts):
            # print("Part", i)
            sent_offset = 0
            cur_part = doc.parts.get(x)
            new_text = cur_part.text.lower()
            new_text = re.sub('\s+', ' ', new_text)
            sentences = new_text.split('. ')
            # TODO use the nltk splitter instead of '. '
            for sent in sentences:
                part_length = len(sent)
                new_text = re.sub('\W+', ' ', sent)
                for i, reg in enumerate(patterns):

                    _lasttime = time.time()  # time start var
                    match = reg.search(new_text)

                    # debug bottleneck patterns
                    _time_current_reg = time.time() - _lasttime  # time end var
                    _pattern_calls += 1  # pattern calls already occured
                    _time_reg_pattern_total += _time_current_reg  # total time spent on searching with patterns
                    if _time_reg_pattern_total > 0:
                        _time_avg_per_pattern = _time_reg_pattern_total / _pattern_calls  # avg spent time per pattern call

                    if _pattern_calls > len(patterns) * 20 and _time_avg_per_pattern * 10000 < _time_current_reg:
                        print("BAD_PATTERN_PERFORMANCE:", _time_avg_per_pattern, _time_current_reg, reg.pattern)
                    if _time_max_pattern < _time_current_reg:
                        _time_max_pattern = _time_current_reg
                        _low_performant_pattern = reg.pattern
                        print(_time_avg_per_pattern, _low_performant_pattern, _time_max_pattern)

                    if reg.pattern == r'(\b\w*\d+\w*\b\s?){1,3} (\b\w+\b\s?){1,4} (\b\w*\d+\w*\b\s?){1,3} (\b\w+\b\s?){1,4} (deletion|deleting|deleted)':
                        if _time_current_reg > _time_avg_per_pattern * 100:
                            print(_time_avg_per_pattern, _time_current_reg)
                            f.write("BAD_PATTERN\n")
                            f.write(sent + "\n")
                            f.write(new_text + "\n")

                    from nala.structures.data import Annotation
                    Annotation.equality_operator = 'exact_or_overlapping'
                    if match:
                        if did in dataset_high_recall.documents:
                            anti_doc = dataset_high_recall.documents.get(did)
                            if not anti_doc.overlaps_with_mention(match.span()):
                                if doc.overlaps_with_mention(match.span()):
                                    TP += 1
                                    f.write("TP: {} {} {}\n".format(sent, match, reg.pattern))
                                else:
                                    FP += 1
                                    f.write("FP: {} {} {}\n".format(sent, match, reg.pattern))
                                break
                    if _lasttime - time.time() > 1:
                        print(i)
                sent_offset += 2 + part_length
            part_offset += sent_offset
        _progress += 1
        _time_progressed = time.time() - _timestart
        _time_per_doc = _time_progressed / _progress
        _time_req_time = _time_per_doc * _length
        _time_eta = _time_req_time - _time_progressed
        print("PROGRESS: {0:.4%} ETA: {1:.2f} secs".format(_progress/_length, _time_eta))
        print('STATS: TP:{}, FP:{}, %containingNLmentions:{:.4%}'.format(TP, FP, TP/(TP + FP)))
