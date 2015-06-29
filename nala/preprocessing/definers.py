import abc
import json
import csv
import re
import os
import requests
import pkg_resources

class NLDefiner:
    """
    Abstract class for determining whether an annotation in the dataset is a natural language (NL) mention.
    Subclasses that inherit this class should:
    * Be named [Name]NLDefiner
    * Implement the abstract method define
    * Set the value
    """
    @abc.abstractmethod
    def define(self, dataset):
        """
        :type dataset: structures.data.Dataset
        """
        return


class InclusiveNLDefiner(NLDefiner):

    def __init__(self, min_length=18):
        self.min_spaces = 3
        self.min_length = min_length

    def define(self, dataset):
        for ann in dataset.annotations():
            if ann.class_id == 'e_2' \
                    and len(ann.text) >= self.min_length \
                    and len(ann.text.split(" ")) > self.min_spaces:
                ann.is_nl = True
            else:
                ann.is_nl = False


class AnkitNLDefiner(NLDefiner):

    def __init__(self, min_length=28):
        self.min_spaces = 4
        self.min_length = min_length

    def define(self, dataset):
        for ann in dataset.annotations():
            if ann.class_id == 'e_2' \
                    and len(ann.text) >= self.min_length \
                    and len(ann.text.split(" ")) > self.min_spaces:
                ann.is_nl = True
            else:
                ann.is_nl = False


class ExclusiveNLDefiner(NLDefiner):

    """docstring for ExclusiveNLDefiner"""

    def __init__(self):
        self.max_spaces = 2
        self.conventions_file = pkg_resources.resource_filename('nala.data', 'regex_st.json')
        self.tmvarregex_file = pkg_resources.resource_filename('nala.data', 'RegEx.NL')

        # read in file regex_st.json into conventions array
        with open(self.conventions_file, 'r') as f:
            conventions = json.loads(f.read())
            self.compiled_regexps_custom = [re.compile(x) for x in conventions]

        # read RegEx.NL (only codes)
        with open(self.tmvarregex_file) as file:
            raw_regexps = list(csv.reader(file, delimiter='\t'))
        regexps = [ x[0] for x in raw_regexps if len(x[0]) < 265 ]
        self.compiled_regexps = [ re.compile(x) for x in regexps]

    def define(self, dataset):
        for ann in dataset.annotations():
            # if ann.class_id == 'e_2':
            #     print(ann.class_id, ann.text)
            if ann.class_id == 'e_2' \
                    and not(len(ann.text.split(" ")) <= self.max_spaces):
                matches_tmvar = [regex.match(ann.text) for regex in self.compiled_regexps]
                matches_custom = [regex.match(ann.text) for regex in self.compiled_regexps_custom]
                if not any(matches_custom) and not any(matches_tmvar):
                    ann.is_nl = True


class TmVarRegexNLDefiner(NLDefiner):
    """
    Definer based just on tmVar regular expressions.

    Algorithm:
    if a mutation annotation matches *any* of the regular expressions
        then it is considered a standard mention
    otherwise we consider it a natural language mention.

    Implements the abstract class NLDefiner.
    """
    def define(self, dataset):
        with open(pkg_resources.resource_filename('nala.data', 'RegEx.NL')) as file:
            regexps = list(csv.reader(file, delimiter='\t'))

        compiled_regexps = []
        for regex in regexps:
            if regex[0].startswith('(?-xism:'):
                try:
                    compiled_regexps.append(re.compile(regex[0].replace('(?-xism:', ''),
                                                       re.VERBOSE | re.IGNORECASE | re.DOTALL | re.MULTILINE))
                except:
                    pass
            else:
                compiled_regexps.append(re.compile(regex[0]))

        for ann in dataset.annotations():
            if ann.class_id == 'e_2':
                matches = [regex.match(ann.text) for regex in compiled_regexps]
                if not any(matches):
                    ann.is_nl = True


class TmVarNLDefiner(NLDefiner):
    """
    Definer based on the complete tmVar NER pipeline.

    Algorithm:
    run tmVar on our dataset and obtain predictions for mutations

    if a mutation annotation is predicted by tmVar
        then it is considered a standard mention
    otherwise we consider it a natural language mention.

    NOTE: Make sure to clear the cache (delete cache.ini file) when running the definer for different corpora.

    Implements the abstract class NLDefiner.
    """
    def define(self, dataset):
        if os.path.isfile('cache.json'):
            tm_var = json.load(open('cache.json'))
        else:
            url_tmvar = 'http://www.ncbi.nlm.nih.gov/CBBresearch/Lu/Demo/RESTful/tmTool.cgi/Mutation/{0}/JSON/'
            url_converter = 'http://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/'

            tm_var = {}
            for pmid, doc in dataset.documents.items():
                # if we have a pmc id instead converted it first to pmid
                if pmid.startswith('PMC'):
                    req = requests.get(url_converter, {'ids': pmid, 'format': 'json'})
                    pmid = req.json()['records'][0]['pmid']

                req = requests.get(url_tmvar.format(pmid))
                try:
                    tm_var[pmid] = req.json()
                except ValueError:
                    pass
            # cache the tmVar annotations so we don't pull them every time
            with open('cache.json', 'w') as file:
                file.write(json.dumps(tm_var, indent=4))

        for doc_id, doc in dataset.documents.items():
            if doc_id in tm_var:
                denotations = tm_var[doc_id]['denotations']
                text = tm_var[doc_id]['text']
                denotations = [text[d['span']['begin']:d['span']['end']] for d in denotations]

                for part_id, part in doc.parts.items():
                    for ann in part.annotations:
                        if ann.class_id == 'e_2' and ann.text not in denotations:
                            ann.is_nl = True
