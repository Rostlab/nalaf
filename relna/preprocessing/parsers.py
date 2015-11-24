import abc
import re
from nala.structures.data import *
from bllipparser import RerankingParser
from nltk.stem.lancaster import LancasterStemmer
from nltk import word_tokenize
from nltk.corpus import stopwords
from progress.bar import Bar

class Parser:
    """
    Abstract class for generating parse tree for each sentence.
    Subclasses that inherit this class should:
    * Be named [Name]ParseTree
    * Implement the abstract method parse
    * Append new items to the list field "edges" of each Part in the dataset
    """

    @abc.abstractmethod
    def parse(self, dataset):
        """
        :type dataset: nala.structures.data.Dataset
        """
        return


class BllipParser(Parser):
    """
    Implementation of the SpaCy English for parsing the each sentence in each
    part separately, finding dependencies, parts of speech tags, lemmas and
    head words for each entity.

    Uses preprocessed text

    :param nbest: the number of parse trees to obtain
    :type nbest: int
    :param overparsing: overparsing determines how much more time the parser
        will spend on a sentence relative to the time it took to find the
        first possible complete parse
    :type overparsing: int
    """
    def __init__(self, nbest=10, overparsing=10, only_parse=False, stop_words=None):
        self.parser = RerankingParser.fetch_and_load('GENIA+PubMed', verbose=False)
        """create a Reranking Parser from BllipParser"""
        self.parser.set_parser_options(nbest=nbest, overparsing=overparsing)
        """set parser options"""
        self.only_parse=only_parse
        """whether features should be used from the BllipParser"""
        self.stemmer = LancasterStemmer()
        """an instance of LancasterStemmer from NLTK"""
        self.stop_words = stop_words
        if self.stop_words is None:
            self.stop_words = stopwords.words('english')

    def parse(self, dataset):
        outer_bar = Bar('Processing [Bllip]', max=len(list(dataset.parts())))
        for part in dataset.parts():
            outer_bar.next()
            if len(part.sentence_parse_trees)>0:
                continue
            for index, sentence in enumerate(part.sentences):
                sentence = [ token.word for token in part.sentences[index] ]
                parse = self.parser.parse(sentence)
                parsed = parse[0]
                part.sentence_parse_trees.append(str(parsed.ptb_parse))
                if not self.only_parse:
                    tokens = parsed.ptb_parse.sd_tokens()
                    for token in tokens:
                        tok = part.sentences[index][token.index-1]
                        is_stop = False
                        if tok.word.lower() in self.stop_words:
                            is_stop = True
                        tok.features = {
                                        'id': token.index-1,
                                        'pos': token.pos,
                                        'lemma': self.stemmer.stem(tok.word),
                                        'is_punct': self._is_punct(tok.word),
                                        'dep': token.deprel,
                                        'is_stop': is_stop,
                                        'dependency_from': None,
                                        'dependency_to': [],
                                        'is_root': False,
                                        }
                        part.tokens.append(tok)
                    for token in tokens:
                        tok = part.sentences[index][token.index-1]
                        self._dependency_path(token, tok, part, index)
            part.percolate_tokens_to_entities()
            part.calculate_token_scores()
            part.set_head_tokens()
        outer_bar.finish()

    def _dependency_path(self, bllip_token, token, part, index):
        if bllip_token.head-1>=0:
            token.features['dependency_from'] = (part.sentences[index][bllip_token.head-1], bllip_token.deprel)
        else:
            token.features['dependency_from'] = (part.sentences[index][token.features['id']], bllip_token.deprel)
        token_from = part.sentences[index][bllip_token.head-1]
        if (bllip_token.index != bllip_token.head):
            token_from.features['dependency_to'].append((token, bllip_token.deprel))
        else:
            token.features['is_root'] = True

    def _is_punct(self, text):
        if text in ['.', ',', '-']:
            return True
        return False


class SpacyParser(Parser):
    """
    Implementation of the SpaCy English for parsing the each sentence in each
    part separately, finding dependencies, parts of speech tags, lemmas and
    head words for each entity.

    :param nlp: an instance of spacy.en.English
    :type nlp: spacy.en.English
    :param constituency_parser: the constituency parser to use to generate
        syntactic (constituency) parse trees. Currently, supports only 'bllip'.
    :type constituency_parser: str
    """
    from spacy.en import English
    from progress.bar import Bar

    def __init__(self, nlp, constituency_parser=True):
        self.nlp = nlp
        """an instance of spacy.en.English"""
        self.constituency_parser = constituency_parser
        """the type of constituency parser to use, current supports only bllip"""
        # TODO SpaCy will soon have it's own constituency parser, integrate that
        # as the default
        if (not isinstance(self.nlp, English)):
            raise TypeError('Not an instance of spacy.en.English')
        # Use the default tokenization done by a call to
        # nala.preprocessing.Tokenizer before.
        old_tokenizer = self.nlp.tokenizer
        self.nlp.tokenizer = lambda string: old_tokenizer.tokens_from_list(self._tokenize(string))
        if self.constituency_parser == True:
            self.parser = BllipParser(only_parse=True)

    def parse(self, dataset):
        """
        :type dataset: nala.structures.data.Dataset
        """
        outer_bar = Bar('Processing [SpaCy]', max=len(list(dataset.parts())))
        for part in dataset.parts():
            sentences = part.get_sentence_string_array()
            for index, sentence in enumerate(sentences):
                doc = self.nlp(sentence)
                for token in doc:
                    tok = part.sentences[index][token.i]
                    tok.features = {
                                    'id': token.i,
                                    'pos': token.tag_,
                                    'dep': token.dep_,
                                    'lemma': token.lemma_,
                                    'prob': token.prob,
                                    'is_punct': token.is_punct,
                                    'is_stop': token.is_stop,
                                    'cluster': token.cluster,
                                    'dependency_from': None,
                                    'dependency_to': [],
                                    'is_root': False,
                                   }
                    part.tokens.append(tok)
                for tok in doc:
                    self._dependency_path(tok, index, part)
            part.percolate_tokens_to_entities()
            part.calculate_token_scores()
            part.set_head_tokens()
            outer_bar.next()
        outer_bar.finish()
        if self.constituency_parser == True:
            self.parser.parse(dataset)

    def _tokenize(self, text):
        return text.split(' ')

    def _dependency_path(self, tok, index, part):
        token = part.sentences[index][tok.i]
        token.features['dependency_from'] = (part.sentences[index][tok.head.i], tok.dep_)
        token_from = part.sentences[index][tok.head.i]
        if (tok.i != tok.head.i):
            token_from.features['dependency_to'].append((token, tok.dep_))
        else:
            token.features['is_root'] = True
