import abc
import json
import re
import time
from preprocessing.definers import InclusiveNLDefiner
from preprocessing.definers import ExclusiveNLDefiner
from preprocessing.spliters import NLTKSplitter
from structures.data import Dataset
from utils.readers import TmVarReader
from utils.tagger import TmVarTagger


class DocumentFilter:
    """
    Abstract base class for filtering out a list of documents (given as a Dataset object)
    according to some criterion.

    Subclasses that inherit this class should:
    * Be named [Name]DocumentFilter
    * Implement the abstract method filter as a generator
    meaning that if some criterion is fulfilled then [yield] that document

    When implementing filter first iterate for each PMID then apply logic to allow chaining of filters.
    """

    @abc.abstractmethod
    def filter(self, documents):
        """
        :type documents: collections.Iterable[nala.structures.data.Document]
        """
        pass


class KeywordsDocumentFilter(DocumentFilter):
    """
    TODO document that this doesn't mean PubMed XML filters
    Filters our documents that do not contain any of the given keywords in any of their parts.
    """
    def __init__(self, keywords=None):
        if not keywords:
            keywords = ('mutation', 'variation', 'substitution', 'insertion', 'deletion', 'snp')
        self.keywords = keywords
        """the keywords which the document should contain"""

    def filter(self, documents):
        """
        :type documents: collections.Iterable[(str, nala.structures.data.Document)]
        """
        for pmid, doc in documents:
            # if any part of the document contains any of the keywords
            # yield that document
            if any(any(keyword in part.text.lower() for keyword in self.keywords)
                   for part in doc.parts.values()):
                yield pmid, doc


class HighRecallRegexDocumentFilter(DocumentFilter):
    """
    Filter that uses regular expression to first get possible natural language mentions in sentences.
    Then each possible nl mention gets compared to tmVar results and Nala predictions. If there is no overlap,
    then this annotation will be considered as nl mention and thus the document will not be filtered out.
    Ever other document gets filtered out at this point.

    Condition for being filtered out:
    Not any(sentence that contains valid nl mention according to this definition)

    tmVar will be used in early stages and discarded as soon as there are no more results, thus gets a parameter.
    """
    def __init__(self, binary_model="nala/data/default_model", override_cache=False, expected_max_results=5, pattern_file='nala/data/nl_patterns.json'):
        self.location_binary_model = binary_model
        """ location where binary model for nala (crfsuite) is saved """
        self.override_cache=override_cache
        """ tmvar results are saved in cache and reused from there.
        this option allows to force requesting results from tmVar online """
        self.expected_maximum_results=expected_max_results
        """ :returns maximum of [x] documents (can be less if not found) """
        # read in nl_patterns
        with open(pattern_file, 'r') as f:
            regexs = json.load(f)
            self.patterns = [re.compile(x) for x in regexs]
            """ compiled regex patterns from pattern_file param to specify custom json file,
             containing regexs for high recall finding of nl mentions. (or sth else) """

    def filter(self, documents):
        """
        :type documents: collections.Iterable[(str, nala.structures.data.Document)]
        """

        dataset = Dataset()
        _list_of_pmids = []
        for pmid, doc in documents:
            dataset.documents[pmid] = doc
            _list_of_pmids.append(pmid)

        # dataset with tmVar
        # note atm just abstracts since full text interface not implemented
        data_tmvar = TmVarTagger().generate_abstracts(_list_of_pmids)
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

        NLTKSplitter().split(dataset)
        for pmid, doc in dataset.documents.items():
            # if any part of the document contains any of the keywords
            # yield that document
            part_offset = 0
            for i, x in enumerate(doc.parts):
                # print("Part", i)
                sent_offset = 0
                cur_part = doc.parts.get(x)
                sentences = cur_part.sentences
                for sent in sentences:
                    sent_length = len(sent)
                    new_text = sent.lower()
                    new_text = re.sub('[\./\\-(){}\[\],%]', '', new_text)
                    new_text = re.sub('\W+', ' ', new_text)
                    for i, reg in enumerate(self.patterns):

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
                            if pmid in data_tmvar.documents:
                                anti_doc = data_tmvar.documents.get(pmid)
                                start = part_offset + sent_offset + match.span()[0]
                                end = part_offset + sent_offset + match.span()[1]
                                if not anti_doc.overlaps_with_mention(start, end):
                                    _e_result = exclusive_definer.define_string(new_text[match.span()[0]:match.span()[1]])
                                    _e_array[_e_result] += 1
                                    _i_result = inclusive_definer.define_string(new_text[match.span()[0]:match.span()[1]])
                                    _i_array[_i_result] += 1
                                    if doc.overlaps_with_mention(match.span()):
                                        TP += 1
                                        print("TP\te{}\ti{}\t{}\t{}\t{}\n".format(_e_result, _i_result, sent, match, reg.pattern))
                                        # _perf_patterns[reg.pattern][0] += 1
                                    else:
                                        FP += 1
                                        print("FP\te{}\ti{}\t{}\t{}\t{}\n".format(_e_result, _i_result, sent, match, reg.pattern))
                                        # _perf_patterns[reg.pattern][1] += 1

                                    # if _perf_patterns[reg.pattern][1] > 0:
                                    #         _perf_patterns[reg.pattern][2] = _perf_patterns[reg.pattern][0] / _perf_patterns[reg.pattern][1]
                                    break
                        if _lasttime - time.time() > 1:
                            print(i)
                    sent_offset += 2 + sent_length  # note why + 2 ?
                part_offset += sent_offset
            _progress += doc.get_size() / _avg_chars_per_doc
            _time_progressed = time.time() - _timestart
            _time_per_doc = _time_progressed / _progress
            _time_req_time = _time_per_doc * _length
            _time_eta = _time_req_time - _time_progressed
            print("PROGRESS: {:.3%} PROGRESS: {:.2f} secs ETA: {:.2f} secs".format(_progress/_length, _time_progressed, _time_eta))
            if TP + FP > 0:
                print('STATS: TP:{}, FP:{}, TP+FP:{} %containingNLmentions:{:.4%}'.format(TP, FP, TP+FP, TP/(TP + FP)))
