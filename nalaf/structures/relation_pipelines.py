from relna.features.context import *
from relna.features.entityhead import *
from relna.features.loctext import *
from relna.features.path import *
from relna.features.sentence import *
from relna.features.ngrams import *
from nalaf.features import FeatureGenerator
from nalaf.structures.data import FeatureDictionary
from nalaf.preprocessing.spliters import Splitter, NLTKSplitter
from nalaf.preprocessing.tokenizers import Tokenizer, TmVarTokenizer
from nalaf.preprocessing.parsers import Parser, SpacyParser
# from nalaf.features import get_spacy_nlp_english
from spacy.en import English
from nalaf.preprocessing.edges import SimpleEdgeGenerator


class RelationExtractionPipeline:
    """
    Prepares an instance of a dataset by executing modules in fixed order.
        * Finally executes each feature generator in the order they were provided

    :param class1: the class of entity1
    :type class1: str
    :param class1: the class of entity2
    :type class1: str
    :param rel_type: the relation type between the two entities
    :type rel_type: str
    :param train: if the mode is training or testing
    :type train: bool
    :param feature_set: the feature_set of the original training data
    :type feature_set: str
    :param feature_generators: one or more modules responsible for generating features
    :type feature_generators: collections.Iterable[FeatureGenerator]
    """

    def __init__(self, class1, class2, rel_type, splitter=None, tokenizer=None, parser=None):
        self.class1 = class1
        self.class2 = class2

        if not splitter:
            splitter = NLTKSplitter()

        if isinstance(splitter, Splitter):
            self.splitter = splitter
        else:
            raise TypeError('not an instance that implements Splitter')

        if not tokenizer:
            tokenizer = TmVarTokenizer()

        if isinstance(tokenizer, Tokenizer):
            self.tokenizer = tokenizer
        else:
            raise TypeError('not an instance that implements Tokenizer')

        self.graphs = {}

        if not parser:
            nlp = English(entity=False)
            parser = SpacyParser(nlp)
        if isinstance(parser, Parser):
            self.parser = parser
        else:
            raise TypeError('not an instance that implements Parser')

        self.edge_generator = SimpleEdgeGenerator(self.class1, self.class2, rel_type)

    def _set_mode(self, train, feature_set, feature_generators=None):
        if feature_generators is None:
            feature_generators = [NamedEntityCountFeatureGenerator(self.class1, feature_set, training_mode=train),
                                  NamedEntityCountFeatureGenerator(self.class2, feature_set, training_mode=train),
                                  BagOfWordsFeatureGenerator(feature_set, training_mode=train),
                                  StemmedBagOfWordsFeatureGenerator(feature_set, training_mode=train),
                                  SentenceFeatureGenerator(feature_set, training_mode=train),
                                  WordFilterFeatureGenerator(feature_set, ['interact', 'bind', 'colocalize'], training_mode=train),
                                  EntityHeadTokenFeatureGenerator(feature_set, training_mode=train),
                                  EntityHeadTokenUpperCaseFeatureGenerator(feature_set, training_mode=train),
                                  EntityHeadTokenDigitsFeatureGenerator(feature_set, training_mode=train),
                                  EntityHeadTokenLetterPrefixesFeatureGenerator(feature_set, training_mode=train),
                                  EntityHeadTokenPunctuationFeatureGenerator(feature_set, training_mode=train),
                                  EntityHeadTokenChainFeatureGenerator(feature_set, training_mode=train),
                                  LinearContextFeatureGenerator(feature_set, training_mode=train),
                                  EntityOrderFeatureGenerator(feature_set, training_mode=train),
                                  LinearDistanceFeatureGenerator(feature_set, training_mode=train),
                                  IntermediateTokensFeatureGenerator(feature_set, training_mode=train),
                                  PathFeatureGenerator(feature_set, self.graphs, training_mode=train),
                                  ProteinWordFeatureGenerator(feature_set, self.graphs, training_mode=train),
                                  LocationWordFeatureGenerator(feature_set, training_mode=train),
                                  FoundInFeatureGenerator(feature_set, training_mode=train),
                                  BiGramFeatureGenerator(feature_set, training_mode=train),
                                  TriGramFeatureGenerator(feature_set, training_mode=train),
                                 ]
        if hasattr(feature_generators, '__iter__'):
            for index, feature_generator in enumerate(feature_generators):
                if not isinstance(feature_generator, FeatureGenerator):
                    raise TypeError('not an instance that implements FeatureGenerator at index {}'.format(index))
                if not feature_generator.training_mode == train:
                    raise ValueError('FeatureGenerator at index {} not set in the correct mode'.format(index))
            self.feature_generators = feature_generators
        elif isinstance(feature_generators, FeatureGenerator):
            if not feature_genenrators.training_mode == train:
                raise ValueError('FeatureGenerator at index not set in the correct mode.')
            else:
                self.feature_generators = [feature_generators]
        else:
            raise TypeError('not an instance or iterable of instances that implements FeatureGenerator')

    def execute(self, dataset, train=False, feature_set=None, feature_generators=None):
        self.feature_set = FeatureDictionary() if feature_set is None else feature_set
        self._set_mode(train, feature_set=self.feature_set, feature_generators=feature_generators)
        try:
            gen = dataset.tokens()
            next(gen)
        except StopIteration:
            self.splitter.split(dataset)
            self.tokenizer.tokenize(dataset)
        self.edge_generator.generate(dataset)
        self.parser.parse(dataset)
        dataset.label_edges()
        for feature_generator in self.feature_generators:
            feature_generator.generate(dataset)
