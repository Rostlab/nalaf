import argparse
import configparser
import sys

from nala.utils.readers import HTMLReader, SETHReader, TmVarReader, VerspoorReader
from nala.preprocessing.spliters import NLTKSplitter
from nala.preprocessing.tokenizers import NLTKTokenizer
from nala.utils.annotation_readers import AnnJsonAnnotationReader, SETHAnnotationReader, VerspoorAnnotationReader
from nala.preprocessing.labelers import BIOLabeler, BIEOLabeler, TmVarLabeler
from nala.preprocessing.definers import TmVarRegexNLDefiner
from nala.preprocessing.definers import ExclusiveNLDefiner, SimpleExclusiveNLDefiner
from nala.preprocessing.definers import TmVarNLDefiner
from nala.preprocessing.definers import InclusiveNLDefiner
from nala.utils.writers import StatsWriter
from nala.features.simple import SimpleFeatureGenerator
from nala.features.tmvar import TmVarFeatureGenerator
from nala.features.window import WindowFeatureGenerator
from nala.learning.crfsuite import CRFSuite

import nala.utils.db_validation as dbcheck
from nala.utils.writers import TagTogFormat
from utils.dataset_selection import RegexSelector, TmVarRegexCombinedSelector

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

        dataset = TmVarReader(html_path).read()

        if not args.quick_nl:
            NLTKSplitter().split(dataset)
            NLTKTokenizer().tokenize(dataset)

        # AnnJsonAnnotationReader(ann_path).annotate(dataset)

        # ttformat = TagTogFormat(to_save_to="demo/output/", dataset=dataset, who="user:verspoor")
        # ttformat.export_html()
        # ttformat.export_ann_json()
        TmVarRegexCombinedSelector().define(dataset)
        exit()

        if args.stats_demo:
            extra_methods = 3
            start_min_length = 18
            end_min_length = 36
            start_counter = start_min_length - extra_methods

            stats = StatsWriter('csvfile.csv', 'graphfile', init_counter=start_counter)

            # tmvar regex
            TmVarRegexNLDefiner().define(dataset)
            tmvarstats = dataset.stats()

            # TODO add param
            if False:
                fullnr = tmvarstats['full_nr']
                abstractnr = tmvarstats['abstract_nr']
                totnr = fullnr + abstractnr
                full_token = tmvarstats['full_tot_token_nr']
                abstract_token = tmvarstats['abstract_tot_token_nr']
                tot_token = full_token + abstract_token
                average_abstract_token = abstract_token / abstractnr
                hypothetical_abstracts_nr = tot_token / average_abstract_token


                print("|Property | Stat |\n|-------|-------|")
                print("|Full documents|", fullnr, "|")
                print("|Abstract documents|", abstractnr, "|")
                print("|Full doc tokens|", full_token, "|")
                print("|Abstract doc tokens|", abstract_token, "|")
                print("|All tokens|", full_token + abstract_token, "|")
                print("|Average tokens per abstract|", "{:.2f}".format(average_abstract_token), "|")
                print("|Hypothetical abstract nr|", "{:.2f}".format(hypothetical_abstracts_nr), "|")

            # for intersection calc
            tmvarmentions = tmvarstats['nl_mention_array']

            stats.addrow(tmvarstats, 'tmVarRegex')
            dataset.clean_nl_definitions()

            # exclusive
            ExclusiveNLDefiner().define(dataset)
            stats.addrow(dataset.stats(), 'Carsten')
            dataset.clean_nl_definitions()

            # tmvar nl
            TmVarNLDefiner().define(dataset)
            stats.addrow(dataset.stats(), 'tmVarComplete')
            dataset.clean_nl_definitions()

            # inclusive
            for i in range(start_min_length, end_min_length + 1):
                print('run', i)
                InclusiveNLDefiner(min_length=i).define(dataset)
                inclusivestats = dataset.stats()
                inclusivementions = inclusivestats['nl_mention_array']
                intersectionset = [ x for x in inclusivementions if x not in tmvarmentions]
                print(intersectionset)
                stats.addrow(dataset.stats(), 'Inclusive_' + str(i))
                dataset.clean_nl_definitions()

            # finally generation of graph itself
            stats.makegraph()

        TmVarRegexNLDefiner().define(dataset)

        if not args.quick_nl:
            print("Labeling")
            TmVarLabeler().label(dataset)
            print("Feature Generation")
            TmVarFeatureGenerator().generate(dataset)
            # print("Window Feature Generation")
            # WindowFeatureGenerator().generate(dataset)

            crf = CRFSuite(crf_path)
            print("CRFstart")
            crf.create_input_file(dataset, 'train')
            crf.learn()
            crf.create_input_file(dataset, 'test')
            crf.tag()

    except (configparser.MissingSectionHeaderError, configparser.NoSectionError, configparser.NoOptionError):
        print(config_ini_error, file=sys.stderr)
