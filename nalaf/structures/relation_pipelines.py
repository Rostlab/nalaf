from nalaf.features import FeatureGenerator
from nalaf.structures.data import FeatureDictionary
from nalaf.features import get_spacy_nlp_english
from nalaf.preprocessing.parsers import SpacyParser
from nalaf.preprocessing.spliters import NLTK_SPLITTER
from nalaf.preprocessing.tokenizers import GenericTokenizer, NLTK_TOKENIZER
from nalaf.preprocessing.edges import SentenceDistanceEdgeGenerator
from nalaf.features.relations.new.sentence import SentenceFeatureGenerator
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

    def __init__(self, class1, class2, rel_type, parser=None, splitter=None, tokenizer=None, edge_generator=None, feature_set=None, feature_generators=None):
        self.class1 = class1
        self.class2 = class2
        self.rel_type = rel_type

        if not parser:
            nlp = get_spacy_nlp_english(load_parser=True)
            parser = SpacyParser(nlp)

        self.parser = parser

        if not splitter:
            # if nlp:  # Spacy parser is used, which includes a sentence splitter
            #     splitter = GenericSplitter(lambda string: (sent.text for sent in nlp(string).sents))
            # else:
            #     splitter = NLTK_SPLITTER
            splitter = NLTK_SPLITTER

        self.splitter = splitter

        if not tokenizer:
            if nlp:  # Spacy parser is used, which includes a tokenizer
                tokenizer = GenericTokenizer(lambda string: (tok.text for tok in nlp.tokenizer(string)))
            else:
                tokenizer = NLTK_TOKENIZER

        self.tokenizer = tokenizer

        self.edge_generator = SentenceDistanceEdgeGenerator(self.class1, self.class2, self.rel_type, distance=0) if edge_generator is None else edge_generator

        self.feature_set = FeatureDictionary() if feature_set is None else feature_set

        self.feature_generators = self._verify_feature_generators(feature_generators) if feature_generators else [
            SentenceFeatureGenerator(f_counts_individual=1),
        ]


    def execute(self, dataset, only_features=False):
        # Note: the order of splitter/tokenizer/edger/parser is important
        # Note: we could avoid the re-splitting & tokenization (see c3d320f08ed8893460d5a68b1b5c87aab6ea0c27)
        #   yet that may later create unforseen problems and re-doing has no significant impact in running time

        start = time.time()

        if not only_features:
            self.splitter.split(dataset)
            self.tokenizer.tokenize(dataset)
            self.parser.parse(dataset)  # Note, the percolate_tokens_to_entities should go before the edge generator due to sentences adjustments
            self.edge_generator.generate(dataset)

        # The labels are always re-generated
        dataset.label_edges()

        for feature_generator in self.feature_generators:
            feature_generator.generate(dataset, self.feature_set, use_gold=self.edge_generator.use_gold, use_pred=self.edge_generator.use_pred)

        end = time.time()
        print_debug("Relation pipeline (only_features: {}), running time: {}".format(only_features, str(end - start)))


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
