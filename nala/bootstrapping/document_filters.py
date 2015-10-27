import abc
from copy import deepcopy
import json
import re
import time
import pkg_resources
from nala.learning.crfsuite import CRFSuite
from nala.learning.postprocessing import PostProcessing
from nala.learning.taggers import CRFSuiteMutationTagger
from nala import print_verbose
from nala.preprocessing.definers import InclusiveNLDefiner
from nala.preprocessing.definers import ExclusiveNLDefiner
from nala.preprocessing.spliters import NLTKSplitter
from nala.structures.data import Dataset
from nala.utils.cache import Cacheable
from nala.utils.readers import TmVarReader
from nala.utils.tagger import TmVarTagger
from nala.structures.dataset_pipelines import PrepareDatasetPipeline


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


class ManualDocumentFilter(DocumentFilter, Cacheable):
    """
    Displays each document to the user on the standard console.
    The user inputs Yes/No as standard input to accept or reject the document.
    """
    def filter(self, documents):
        """
        :type documents: collections.Iterable[(str, nala.structures.data.Document)]
        """
        for pmid, doc in documents:
            # if we can't find it in the cache
            # ask the user and save it to the cache
            if pmid not in self.cache:
                answer = input('{}\nDo you accept this document?\n'.
                               format('\n'.join(part.text for part in doc.parts.values())))
                self.cache[pmid] = answer.lower() in ['yes', 'y']

            if self.cache[pmid]:
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
    def __init__(self, binary_model="nala/data/default_model", override_cache=False, expected_max_results=10, pattern_file_name=None, crfsuite_path=None):
        self.location_binary_model = binary_model
        """ location where binary model for nala (crfsuite) is saved """
        self.override_cache=override_cache
        """ tmvar results are saved in cache and reused from there.
        this option allows to force requesting results from tmVar online """
        self.expected_maximum_results=expected_max_results
        """ :returns maximum of [x] documents (can be less if not found) """
        self.crfsuite_path=crfsuite_path
        """ crfsuite path"""
        # read in nl_patterns
        if not pattern_file_name:
            pattern_file_name = pkg_resources.resource_filename('nala.data', 'dict_nl_words.json')

        with open(pattern_file_name, 'r') as f:
            regexs = json.load(f)
            self.patterns = [re.compile(x) for x in regexs]
            """ compiled regex patterns from pattern_file param to specify custom json file,
             containing regexs for high recall finding of nl mentions. (or sth else) """

    def filter(self, documents, min_found=10):
        """
        :type documents: collections.Iterable[(str, nala.structures.data.Document)]
        """

        _progress = 1
        _timestart = time.time()

        _time_avg_per_pattern = 0
        _pattern_calls = 0
        _time_reg_pattern_total = 0
        _time_max_pattern = 0
        _low_performant_pattern = ""

        # NLDefiners init
        exclusive_definer = ExclusiveNLDefiner()
        _e_array = [0, 0, 0]
        inclusive_definer = InclusiveNLDefiner()
        _i_array = [0, 0]

        last_found = 0
        if self.crfsuite_path:
            crfsuite = CRFSuite(self.crfsuite_path)
            crfsuitetagger = CRFSuiteMutationTagger(['e_2'], crf_suite=crfsuite, model_file=self.location_binary_model)
        for pmid, doc in documents:
            # if any part of the document contains any of the keywords
            # yield that document
            part_offset = 0
            data_tmp = Dataset()
            data_tmp.documents[pmid] = doc
            data_nala = deepcopy(data_tmp)
            NLTKSplitter().split(data_tmp)
            data_tmvar = TmVarTagger().generate_abstracts([pmid])
            if self.crfsuite_path:
                PrepareDatasetPipeline().execute(data_nala)
                crfsuitetagger.tag(data_nala)
                PostProcessing().process(data_nala)
            positive_sentences = 0
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

                    found_in_sentence = False

                    for i, reg in enumerate(self.patterns):
                        _lasttime = time.time()  # time start var
                        match = reg.search(new_text)

                        # debug bottleneck patterns
                        _time_current_reg = time.time() - _lasttime  # time end var
                        _pattern_calls += 1  # pattern calls already occured
                        _time_reg_pattern_total += _time_current_reg  # total time spent on searching with patterns
                        if _time_reg_pattern_total > 0:
                            _time_avg_per_pattern = _time_reg_pattern_total / _pattern_calls  # avg spent time per pattern call
                        # todo create pattern performance eval for descending amount of recognized patterns
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
                                if self.crfsuite_path:
                                    nala_doc = data_nala.documents.get(pmid)
                                start = part_offset + sent_offset + match.span()[0]
                                end = part_offset + sent_offset + match.span()[1]
                                # print("TmVar is not overlapping?:", not anti_doc.overlaps_with_mention(start, end))
                                # print(not nala_doc.overlaps_with_mention(start, end, annotated=False))
                                if self.crfsuite_path:
                                    if not anti_doc.overlaps_with_mention(start,
                                                                          end) \
                                            and not nala_doc.overlaps_with_mention(start, end, annotated=False):
                                        _e_result = exclusive_definer.define_string(
                                            new_text[match.span()[0]:match.span()[1]])
                                        _e_array[_e_result] += 1
                                        _i_result = inclusive_definer.define_string(
                                            new_text[match.span()[0]:match.span()[1]])
                                        _i_array[_i_result] += 1
                                        # todo write to file param + saving to manually annotate and find tp + fp for performance eval on each pattern
                                        # print("e{}\ti{}\t{}\t{}\t{}\n".format(_e_result, _i_result, sent, match, reg.pattern))

                                        last_found += 1
                                        found_in_sentence = True
                                else:
                                    if not anti_doc.overlaps_with_mention(start, end):
                                        _e_result = exclusive_definer.define_string(
                                            new_text[match.span()[0]:match.span()[1]])
                                        _e_array[_e_result] += 1
                                        _i_result = inclusive_definer.define_string(
                                            new_text[match.span()[0]:match.span()[1]])
                                        _i_array[_i_result] += 1
                                        # todo write to file param + saving to manually annotate and find tp + fp for performance eval on each pattern
                                        # print("e{}\ti{}\t{}\t{}\t{}\n".format(_e_result, _i_result, sent, match, reg.pattern))
                                        last_found += 1
                                        found_in_sentence = True

                        if _lasttime - time.time() > 1:
                            print(i)
                    sent_offset += 2 + sent_length

                    # for per sentence positives
                    if found_in_sentence:
                        positive_sentences += 1

                part_offset += sent_offset
            if positive_sentences > min_found:
                _progress += 1
            _time_progressed = time.time() - _timestart
            _time_per_doc = _time_progressed / _progress
            print_verbose("PROGRESS: {:.2f} secs ETA per one positive document: {:.2f} secs".format(_time_progressed, _time_per_doc))
            if positive_sentences > min_found:
                last_found = 0
                print_verbose('YEP')
                print('Found')
                yield pmid, doc
            else:
                print_verbose(pmid, "contains either no or only a few suitable annotations")
                print_verbose('NOPE')
                print('Not Found')
