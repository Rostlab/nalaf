import argparse
import os
from nala.utils.readers import TextFilesReader
from nala.utils.readers import StringReader
from nala.utils.writers import ConsoleWriter
from nala.learning.crfsuite import CRFSuite
from nala.preprocessing.spliters import NLTKSplitter
from nala.preprocessing.tokenizers import TmVarTokenizer
from nala.features.simple import SimpleFeatureGenerator
from nala.features.stemming import PorterStemFeatureGenerator
from nala.features.tmvar import TmVarFeatureGenerator
from nala.features.tmvar import TmVarDictionaryFeatureGenerator
from nala.features.window import WindowFeatureGenerator
from nala.learning.postprocessing import PostProcessing

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='A simple demo for using the nala pipeline for prediction')

    parser.add_argument('-c', '--crf_suite_dir', help='path to the directory containing the crfsuite executable',
                        required=True)

    parser.add_argument('--color', help='uses color for highlighting predictions if supported '
                                        'otherwise prints them in new line',
                        action='store_true', default=True, dest='color')
    parser.add_argument('--no_color', help='prints predictions in new line',
                        action='store_false', dest='color')

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument('-s', '--string', help='string you want to predict for')
    group.add_argument('-d', '--dir_or_file', help='directory or file you want to predict for')
    args = parser.parse_args()

    if args.string:
        dataset = StringReader(args.string).read()
    elif os.path.exists(args.dir_or_file):
        dataset = TextFilesReader(args.dir_or_file).read()
    else:
        raise FileNotFoundError('directory or file "{}" does not exist'.format(args.dir_or_file))

    #TODO include default_model & example.txt under resources/
    default_model_path = os.path.join(os.getcwd(), 'default_model')
    if not os.path.exists(default_model_path):
        raise FileNotFoundError('default_model is missing')

    #TODO refactor to create pipeline (object) which accepts sentenceSplitter, tokenizer, labeler, list of feature generators, and other options such as usePreprocessing, ...
    #TODO 2 pipelines, `PrepareDataset` & `UseDataset`
    #TODO if possible, change parts & sentences & tokens from List to Tuple

    # split and tokenize
    NLTKSplitter().split(dataset)
    TmVarTokenizer().tokenize(dataset)

    # generate features
    SimpleFeatureGenerator().generate(dataset)
    PorterStemFeatureGenerator().generate(dataset)
    TmVarFeatureGenerator().generate(dataset)
    TmVarDictionaryFeatureGenerator().generate(dataset)

    window_include_list = ['pattern0[0]', 'pattern1[0]', 'pattern2[0]', 'pattern3[0]', 'pattern4[0]', 'pattern5[0]',
    'pattern6[0]', 'pattern7[0]', 'pattern8[0]', 'pattern9[0]', 'pattern10[0]', 'word[0]', 'stem[0]']

    # generate features in a window
    WindowFeatureGenerator(template=(-3, -2, -1, 1, 2, 3), include_list=window_include_list).generate(dataset)

    # get the predictions
    crf = CRFSuite(args.crf_suite_dir)
    crf.create_input_file(dataset, 'predict')
    crf.tag('-m {} -i predict > output.txt'.format(default_model_path))
    crf.read_predictions(dataset)

    PostProcessing().process(dataset)
    ConsoleWriter(args.color).write(dataset)
