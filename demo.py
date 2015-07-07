import argparse
import configparser
import sys

from nala.utils.readers import HTMLReader
from nala.preprocessing.spliters import NLTKSplitter
from nala.preprocessing.tokenizers import NLTKTokenizer
from nala.utils.annotation_readers import AnnJsonAnnotationReader
from nala.preprocessing.labelers import BIOLabeler
from nala.preprocessing.definers import TmVarRegexNLDefiner
from nala.preprocessing.definers import ExclusiveNLDefiner
from nala.preprocessing.definers import TmVarNLDefiner
from nala.preprocessing.definers import InclusiveNLDefiner
from nala.utils.writers import StatsWriter
from nala.features.simple import SimpleFeatureGenerator
from nala.learning.crfsuite import CRFSuite

import nala.utils.db_validation as dbcheck

if __name__ == "__main__":
    config_ini_help = 'Configuration file containing the paths to the dataset, annotations and crfsuite executable. ' \
                      'Defaults to config.ini.'
    config_ini_error = """
    The configuration file doesn't have the expected format.
    The file should have the following format:
    [paths]
    html_path = [path to the directory containing the articles in html format]
    ann_path = [path to the directory containing the annotations of the articles in ann.json format]
    crf_path = [path to the directory containing the crfsuite executable]
    """

    config_checkdb_help = """
    To Check for integrity of html with ann.json files (offsets, existing ids).
    Only check for integrity and then exit.
    """

    config_stats_demo_help="""
    Generate a graph for several NL Definitions.
    """

    parser = argparse.ArgumentParser(description='A simple demo of using the nala pipeline')
    parser.add_argument('-c', '--config', type=argparse.FileType('r'), help=config_ini_help, default='config.ini')
    parser.add_argument('-db', '--check-db', action='store_true', help=config_checkdb_help)
    parser.add_argument('--stats-demo', action='store_true', help=config_stats_demo_help)
    parser.add_argument('-qnl', '--quick-nl', action='store_true', help='Quick Run w/o tokenizer, labeler and crfsuite')
    args = parser.parse_args()

    config = configparser.ConfigParser()
    try:
        config.read_file(args.config)

        html_path = config.get('paths', 'html_path')
        ann_path = config.get('paths', 'ann_path')
        crf_path = config.get('paths', 'crf_path')

        # if only to check db then do that...
        if args.check_db:
            dbcheck.main(html_path=html_path, ann_path=ann_path)
            exit()

        dataset = HTMLReader(html_path).read()

        if not args.quick_nl:
            NLTKSplitter().split(dataset)
            NLTKTokenizer().tokenize(dataset)

        AnnJsonAnnotationReader(ann_path).annotate(dataset)

        if args.stats_demo:
            extra_methods = 3
            start_min_length = 18
            end_min_length = 36
            start_counter = start_min_length - extra_methods

            stats = StatsWriter('csvfile.csv', 'graphfile', init_counter=start_counter)

            # tmvar regex
            TmVarRegexNLDefiner().define(dataset)
            tmvarstats = dataset.stats()
            tmvarmentions = tmvarstats['nl_mention_array']
            stats.addrow(tmvarstats, 'tmVarRegex')
            dataset.cleannldefinitions()

            # exclusive
            ExclusiveNLDefiner().define(dataset)
            stats.addrow(dataset.stats(), 'Carsten')
            dataset.cleannldefinitions()

            # tmvar nl
            TmVarNLDefiner().define(dataset)
            stats.addrow(dataset.stats(), 'tmVarComplete')
            dataset.cleannldefinitions()

            # inclusive
            for i in range(start_min_length, end_min_length + 1):
                print('run', i)
                InclusiveNLDefiner(min_length=i).define(dataset)
                inclusivestats = dataset.stats()
                inclusivementions = inclusivestats['nl_mention_array']
                intersectionset = [ x for x in inclusivementions if x not in tmvarmentions]
                print(intersectionset)
                stats.addrow(dataset.stats(), 'Inclusive_' + str(i))
                dataset.cleannldefinitions()

            # finally generation of graph itself
            stats.makegraph()

        TmVarRegexNLDefiner().define(dataset)

        if not args.quick_nl:
            BIOLabeler().label(dataset)
            SimpleFeatureGenerator().generate(dataset)

            crf = CRFSuite(crf_path)
            crf.create_input_file(dataset, 'train')
            crf.train()
            crf.create_input_file(dataset, 'test')
            crf.test()

    except (configparser.MissingSectionHeaderError, configparser.NoSectionError, configparser.NoOptionError):
        print(config_ini_error, file=sys.stderr)
