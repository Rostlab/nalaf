import argparse
import configparser
import sys

from nala.utils.readers import HTMLReader
from nala.preprocessing.spliters import NLTKSplitter
from nala.preprocessing.tokenizers import NLTKTokenizer
from nala.utils.annotation_readers import AnnJsonAnnotationReader
from nala.preprocessing.labelers import SimpleLabeler
from nala.preprocessing.definers import TmVarRegexNLDefiner
from nala.features.simple import SimpleFeatureGenerator
from nala.learning.crfsuite import CRFSuite

if __name__ == "__main__":
    config_ini_help = 'Configuration file containing the paths to the dataset, annotations and crfsuite executable.'
    config_ini_error = """
    The configuration file doesn't have the expected format.
    The file should have the following format:
    [paths]
    html_path = [path to the directory containing the articles in html format]
    ann_path = [path to the directory containing the annotations of the articles in ann.json format]
    crf_path = [path to the directory containing the crfsuite executable]
    """

    parser = argparse.ArgumentParser(description='A simple demo of using the nala pipeline')
    parser.add_argument('-c', '--config', type=argparse.FileType('r'), help=config_ini_help, required=True)
    args = parser.parse_args()

    config = configparser.ConfigParser()
    try:
        config.read_file(args.config)

        html_path = config.get('paths','html_path')
        ann_path = config.get('paths','ann_path')
        crf_path = config.get('paths','crf_path')

        dataset = HTMLReader(html_path).read()

        NLTKSplitter().split(dataset)
        NLTKTokenizer().tokenize(dataset)

        AnnJsonAnnotationReader(ann_path).annotate(dataset)
        TmVarRegexNLDefiner().define(dataset)

        SimpleLabeler().label(dataset)
        SimpleFeatureGenerator().generate(dataset)

        crf = CRFSuite(crf_path)
        crf.create_input_file(dataset, 'train')
        crf.train()
        crf.create_input_file(dataset, 'test')
        crf.test()

    except (configparser.MissingSectionHeaderError, configparser.NoSectionError, configparser.NoOptionError):
        print(config_ini_error, file=sys.stderr)








