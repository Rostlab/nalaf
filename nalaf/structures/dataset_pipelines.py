from nalaf.features import FeatureGenerator
from nalaf.features.simple import SimpleFeatureGenerator
from nalaf.features.stemming import PorterStemFeatureGenerator
from nalaf.features.window import WindowFeatureGenerator
from nalaf.preprocessing.spliters import NLTKSplitter, Splitter
from nalaf.preprocessing.tokenizers import TmVarTokenizer, Tokenizer
from nalaf import print_verbose


class PrepareDatasetPipeline:
    """
    Prepares an instance of a dataset by executing modules in fixed order.
        * First executes the sentence splitter
        * Next executes the tokenizer
        * Finally executes each feature generator in the order they were provided

    :type splitter: nalaf.structures.data.Splitter
    :param splitter: the module responsible for splitting the text into sentences
    :type tokenizer: nalaf.structures.data.Tokenizer
    :param tokenizer: the module responsible for splitting the sentences into tokens
    :type feature_generators: collections.Iterable[FeatureGenerator]
    :param feature_generators: one or more modules responsible for generating features
    """

    def __init__(self, splitter=None, tokenizer=None, feature_generators=None):
        if not splitter:
            splitter = NLTKSplitter()
        if not tokenizer:
            tokenizer = TmVarTokenizer()
        if feature_generators is None:
            feature_generators = [SimpleFeatureGenerator(), PorterStemFeatureGenerator(),
                                  WindowFeatureGenerator((-3, -2, -1, 1, 2, 3), ['stem[0]'])]

        if isinstance(splitter, Splitter):
            self.splitter = splitter
        else:
            raise TypeError('not an instance that implements Splitter')

        if isinstance(tokenizer, Tokenizer):
            self.tokenizer = tokenizer
        else:
            raise TypeError('not an instance that implements Tokenizer')

        if hasattr(feature_generators, '__iter__'):
            for index, feature_generator in enumerate(feature_generators):
                if not isinstance(feature_generator, FeatureGenerator):
                    raise TypeError('not an instance that implements FeatureGenerator at index {}'.format(index))
            self.feature_generators = feature_generators
        elif isinstance(feature_generators, FeatureGenerator):
            self.feature_generators = [feature_generators]
        else:
            raise TypeError('not an instance or iterable of instances that implements FeatureGenerator')


    def execute(self, dataset):
        """
        :type dataset: nalaf.structures.data.Dataset()
        """

        self.splitter.split(dataset)
        self.tokenizer.tokenize(dataset)
        for feature_generator in self.feature_generators:
            print_verbose('Apply feature generator:', type(feature_generator))
            feature_generator.generate(dataset)


    def serialize(self, dataset, to_file=None):
        """
        :type dataset: nalaf.structures.data.Dataset()
        """

        types = [(type(self.splitter), self.splitter.__dict__),
                 (type(self.tokenizer), self.tokenizer.__dict__)]

        for feature_generator in self.feature_generators:
            types.append((type(feature_generator), feature_generator.__dict__))

        features = sorted(set(feature_name for token in dataset.tokens() for feature_name in token.features.keys()))

        from nalaf.utils.helpers import find_current_git_ref
        current_ref = find_current_git_ref()

        if to_file:
            with open(to_file, 'w') as file:
                file.write('git ref:{}'.format(current_ref))
                file.write('\nINSTANCES USED\n')
                file.writelines('\n'.join(repr(x) for x in types))
                file.write('\nFEATURES USED\n')
                file.writelines('\n'.join(repr(x) for x in features))

        return types, features, find_current_git_ref()
