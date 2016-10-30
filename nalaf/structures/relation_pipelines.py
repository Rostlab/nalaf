from nalaf.features import FeatureGenerator
from nalaf.structures.data import FeatureDictionary
from nalaf.preprocessing.spliters import Splitter, NLTKSplitter
from nalaf.preprocessing.tokenizers import Tokenizer, TmVarTokenizer
from nalaf.preprocessing.parsers import Parser, SpacyParser
# from nalaf.features import get_spacy_nlp_english
from spacy.en import English
from nalaf.preprocessing.edges import SimpleEdgeGenerator
from nalaf.features.relations.sentence import NamedEntityCountFeatureGenerator
from nalaf import print_debug
import time


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

    def __init__(self, class1, class2, rel_type, splitter=None, tokenizer=None, parser=None, edge_generator=None, feature_set=None, feature_generators=None):
        self.class1 = class1
        self.class2 = class2
        self.rel_type = rel_type

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

        if not parser:
            start = time.time()
            print_debug("(relation_pipelines.py) Spacy NLP English: INIT START")
            nlp = English(entity=False)
            parser = SpacyParser(nlp)
            print_debug("(relation_pipelines.py) Spacy NLP English: INIT END", (time.time() - start))
        if isinstance(parser, Parser):
            self.parser = parser
        else:
            raise TypeError('not an instance that implements Parser')

        self.edge_generator = SimpleEdgeGenerator(self.class1, self.class2, self.rel_type) if edge_generator is None else edge_generator

        self.feature_set = FeatureDictionary() if feature_set is None else feature_set

        self.feature_generators = self._verify_feature_generators(feature_generators) if feature_generators else [
            NamedEntityCountFeatureGenerator(self.class1),
            NamedEntityCountFeatureGenerator(self.class2)
        ]


    def execute(self, dataset, train):
        try:
            gen = dataset.tokens()
            next(gen)
        except StopIteration:
            self.splitter.split(dataset)
            self.tokenizer.tokenize(dataset)

        self.edge_generator.generate(dataset)
        dataset.label_edges()
        self.parser.parse(dataset)

        for feature_generator in self.feature_generators:
            feature_generator.generate(dataset, self.feature_set, train)


    def _verify_feature_generators(self, feature_generators):
        if hasattr(feature_generators, '__iter__'):
            for index, feature_generator in enumerate(feature_generators):
                if not isinstance(feature_generator, FeatureGenerator):
                    raise TypeError('not an instance that implements FeatureGenerator at index {}'.format(index))

            return feature_generators

        elif isinstance(feature_generators, FeatureGenerator):
            return [feature_generators]

        else:
            raise TypeError('not an instance or iterable of instances that implements FeatureGenerator')
