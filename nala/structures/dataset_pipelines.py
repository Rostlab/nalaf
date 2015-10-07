from nala.features import FeatureGenerator
from nala.features.simple import SimpleFeatureGenerator
from nala.features.stemming import PorterStemFeatureGenerator
from nala.features.tmvar import TmVarFeatureGenerator, TmVarDictionaryFeatureGenerator
from nala.features.window import WindowFeatureGenerator
from nala.preprocessing.spliters import NLTKSplitter, Splitter
from nala.preprocessing.tokenizers import TmVarTokenizer, Tokenizer

__author__ = 'Aleksandar'


class PrepareDatasetPipeline:
    """
    Prepares an instance of a dataset by executing modules in fixed order.
        * First executes the sentence splitter
        * Next executes the tokenizer
        * Finally executes each feature generator in the order they were provided

    :type splitter: nala.structures.data.Splitter
    :param splitter: the module responsible for splitting the text into sentences
    :type tokenizer: nala.structures.data.Tokenizer
    :param tokenizer: the module responsible for splitting the sentences into tokens
    :type feature_generators: collections.Iterable[FeatureGenerator]
    :param feature_generators: one or more modules responsible for generating features
    """

    def __init__(self, splitter=None, tokenizer=None, feature_generators=None):
        if not splitter:
            splitter = NLTKSplitter()
        if not tokenizer:
            tokenizer = TmVarTokenizer()
        if not feature_generators:
            include = ['pattern0[0]', 'pattern1[0]', 'pattern2[0]', 'pattern3[0]', 'pattern4[0]', 'pattern5[0]',
                       'pattern6[0]', 'pattern7[0]', 'pattern8[0]', 'pattern9[0]', 'pattern10[0]', 'word[0]', 'stem[0]']
            feature_generators = [SimpleFeatureGenerator(), PorterStemFeatureGenerator(), TmVarFeatureGenerator(),
                                  TmVarDictionaryFeatureGenerator(),
                                  WindowFeatureGenerator(template=(-3, -2, -1, 1, 2, 3), include_list=include)]

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
        :type dataset: nala.structures.data.Dataset()
        """
        self.splitter.split(dataset)
        self.tokenizer.tokenize(dataset)
        for feature_generator in self.feature_generators:
            feature_generator.generate(dataset)

    def serialize(self, dataset, to_file=None):
        """
        :type dataset: nala.structures.data.Dataset()
        """
        types = [(type(self.splitter), self.splitter.__dict__),
                 (type(self.tokenizer), self.tokenizer.__dict__)]

        for feature_generator in self.feature_generators:
            types.append((type(feature_generator), feature_generator.__dict__))

        features = sorted(set(feature_name for token in dataset.tokens() for feature_name in token.features.keys()))

        from nala.utils.helpers import find_current_git_ref
        current_ref = find_current_git_ref()

        if to_file:
            with open(to_file, 'w') as file:
                file.write('git ref:{}'.format(current_ref))
                file.write('\nINSTANCES USED\n')
                file.writelines('\n'.join(repr(x) for x in types))
                file.write('\nFEATURES USED\n')
                file.writelines('\n'.join(repr(x) for x in features))

        return types, features, find_current_git_ref()
