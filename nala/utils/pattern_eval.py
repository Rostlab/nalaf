import json
import re
import time
from nala.preprocessing.definers import ExclusiveNLDefiner, InclusiveNLDefiner
from nala.utils.readers import TmVarReader


def pattern_stats(dataset):
    """
    Testing Ground Carsten - High Recall Patterns creation method development method ported here.
    :type nala.structures.Dataset: dataset to perform pattern evaluation on (must include annotations)
    :return: nothing (print statements for the moment)
    """
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

    # for ann in dataset.annotations():
    #     if ann.subclass == 1 or ann.subclass == 2:
    #         new_text = ann.text.lower()
    #         for reg in compiled_regexs:
    #             new_text = reg.sub('_TT_', new_text)
    #         # re.sub('\\b\\d+\\b]', '_NR_', new_text)
    #         new_text = re.sub('\\b\\w*\\d+\\w*\\b', '_CODE_', new_text)
    #         new_text = nr_word_regex.sub('_TT_', new_text)
    #         new_text = aa_short_regex.sub('_AA_', new_text)
    #         new_text = aa_long_regex.sub('_AA_', new_text)
    #         new_text = bp_code.sub('_TT_', new_text)
    #         new_text = re.sub('\\W', ' ', new_text)
    #         # new_text = re.sub('\\b(\\w{1,3})\\b', '_TT_', new_text)
    #
    #         wordlist.extend(new_text.split(' '))
    #         # print(new_text)
    #         nl_annotations.append(new_text)
    #
    # wordset = set(wordlist)
    # wordlist = sorted(list(wordset))
    # print(json.dumps(wordlist, indent=2, sort_keys=True))
    # print(json.dumps(nl_annotations, indent=2, sort_keys=True))

    # todo provide method to create new pattern on an automated base
    # read in nl_patterns
    with open('nala/data/nl_patterns.json', 'r') as f:
        regexs = json.load(f)

    patterns = [re.compile(x) for x in regexs]

    # f-measure pattern-based
    _perf_patterns = {}
    for reg in patterns:
        _perf_patterns[reg.pattern] = [0, 0, -1]

    # check for annotations

    # for part in dataset.parts():
    #     print(part.text)

    # dataset with tmVar
    # TODO change if idp4 then those results otherwise use tmvartagger and caching
    dataset_high_recall = TmVarReader('resources/corpora/idp4/pubtator_tmvar.txt').read()
    TP = 0
    FP = 0
    _length = len(dataset.documents.keys())
    _progress = 0
    _timestart = time.time()

    _time_avg_per_pattern = 0
    _pattern_calls = 0
    _time_reg_pattern_total = 0
    _time_max_pattern = 0
    _low_performant_pattern = ""
    _avg_chars_per_doc = dataset.get_size_chars() / len(dataset.documents.keys())

    # NLDefiners init
    exclusive_definer = ExclusiveNLDefiner()
    _e_array = [0, 0, 0]
    inclusive_definer = InclusiveNLDefiner()
    _i_array = [0, 0]

    # todo param file to save to
    with open('results/testing_ground_carsten.txt', 'w', encoding='utf-8') as f:
        for did, doc in dataset.documents.items():
            part_offset = 0
            for i, x in enumerate(doc.parts):
                # print("Part", i)
                sent_offset = 0
                cur_part = doc.parts.get(x)
                sentences = cur_part.sentences
                # new_text = cur_part.text.lower()
                # new_text = re.sub('\s+', ' ', new_text)
                # sentences = new_text.split('. ')
                for sent in sentences:
                    sent_len = len(sent)
                    new_text = sent.lower()
                    new_text = re.sub('[\./\\-(){}\[\],%]', '', new_text)
                    new_text = re.sub('\W+', ' ', new_text)
                    for i, reg in enumerate(patterns):

                        _lasttime = time.time()  # time start var
                        match = reg.search(new_text)

                        # debug bottleneck patterns
                        _time_current_reg = time.time() - _lasttime  # time end var
                        _pattern_calls += 1  # pattern calls already occured
                        _time_reg_pattern_total += _time_current_reg  # total time spent on searching with patterns
                        if _time_reg_pattern_total > 0:
                            _time_avg_per_pattern = _time_reg_pattern_total / _pattern_calls  # avg spent time per pattern call

                        # if _pattern_calls > len(patterns) * 20 and _time_avg_per_pattern * 10000 < _time_current_reg:
                        #     print("BAD_PATTERN_PERFORMANCE:", _time_avg_per_pattern, _time_current_reg, reg.pattern)
                        # if _time_max_pattern < _time_current_reg:
                        #     _time_max_pattern = _time_current_reg
                        #     _low_performant_pattern = reg.pattern
                        #     print(_time_avg_per_pattern, _low_performant_pattern, _time_max_pattern)

                        # if reg.pattern == r'(\b\w*\d+\w*\b\s?){1,3} (\b\w+\b\s?){1,4} (\b\w*\d+\w*\b\s?){1,3} (\b\w+\b\s?){1,4} (deletion|deleting|deleted)':
                        #     if _time_current_reg > _time_avg_per_pattern * 10:
                        #         # print(_time_avg_per_pattern, _time_current_reg)
                        #         f.write("BAD_PATTERN\n")
                        #         f.write(sent + "\n")
                        #         f.write(new_text + "\n")

                        if match:
                            if did in dataset_high_recall.documents:
                                anti_doc = dataset_high_recall.documents.get(did)
                                start = part_offset + sent_offset + match.span()[0]
                                end = part_offset + sent_offset + match.span()[1]
                                if not anti_doc.overlaps_with_mention(start, end):
                                    _e_result = exclusive_definer.define_string(new_text[match.span()[0]:match.span()[1]])
                                    _e_array[_e_result] += 1
                                    _i_result = inclusive_definer.define_string(new_text[match.span()[0]:match.span()[1]])
                                    _i_array[_i_result] += 1
                                    if doc.overlaps_with_mention(start, end):
                                        TP += 1
                                        f.write("{}\tTP\te{}\ti{}\t{}\t{}\t{}\n".format(did, _e_result, _i_result, sent, match, reg.pattern))
                                        _perf_patterns[reg.pattern][0] += 1
                                    else:
                                        FP += 1
                                        f.write("{}\tFP\te{}\ti{}\t{}\t{}\t{}\n".format(did, _e_result, _i_result, sent, match, reg.pattern))
                                        _perf_patterns[reg.pattern][1] += 1

                                    if _perf_patterns[reg.pattern][1] > 0:
                                            _perf_patterns[reg.pattern][2] = _perf_patterns[reg.pattern][0] / _perf_patterns[reg.pattern][1]
                        if _lasttime - time.time() > 1:
                            print(i)
                    sent_offset += 2 + sent_len
                part_offset += sent_offset
            _progress += doc.get_size() / _avg_chars_per_doc
            _time_progressed = time.time() - _timestart
            _time_per_doc = _time_progressed / _progress
            _time_req_time = _time_per_doc * _length
            _time_eta = _time_req_time - _time_progressed
            print("PROGRESS: {:.3%} PROGRESS: {:.2f} secs ETA: {:.2f} secs".format(_progress/_length, _time_progressed, _time_eta))
            if TP + FP > 0:
                print('STATS: TP:{}, FP:{}, TP+FP:{} %containingNLmentions:{:.4%}'.format(TP, FP, TP+FP, TP/(TP + FP)))

    print("Exclusive Definer:", _e_array)
    print("Inclusive Definer:", _i_array)

    for key, value in _perf_patterns.items():
        if value[2] != -1:
            print(value, key)

    # todo save performances to file
    # print(json.dumps(_perf_patterns, indent=3, sort_keys=True))
