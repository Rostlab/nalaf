import argparse
import configparser
import json
import os
import sys
import math
from nala.bootstrapping.iteration import Iteration

from nala.utils.readers import HTMLReader, SETHReader, TmVarReader, VerspoorReader
from nala.preprocessing.spliters import NLTKSplitter
from nala.preprocessing.tokenizers import NLTKTokenizer
from nala.utils.annotation_readers import AnnJsonAnnotationReader, SETHAnnotationReader
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


def print_stats(data_stats):
    abstractnr = data_stats['abstract_nr']
    fullnr = data_stats['full_nr']
    totnr = fullnr + abstractnr

    abstract_token = data_stats['abstract_tot_token_nr']
    full_token = data_stats['full_tot_token_nr']
    tot_token = full_token + abstract_token

    if abstractnr == 0:
        average_abstract_token = abstract_token / totnr
    else:
        average_abstract_token = abstract_token / abstractnr
    hypothetical_abstracts_nr = tot_token / average_abstract_token

    nl_mention_nr = data_stats['nl_mention_nr']
    tot_mention_nr = data_stats['tot_mention_nr']
    rationl = nl_mention_nr / tot_mention_nr

    abstract_nl_mention_nr = data_stats['abstract_nl_token_nr']
    nl_norm_abstract = abstract_nl_mention_nr / abstract_token
    try:
        full_nl_mention_nr = data_stats['full_nl_token_nr']
        nl_norm_full = full_nl_mention_nr / full_token
        ratio_abstract_full = nl_norm_abstract / nl_norm_full
    except ZeroDivisionError:
        ratio_abstract_full = 0
        nl_norm_full = 0

    format_number_string = "| {:^28} | {:^14} |"
    format_number_digit = "| {:28} | {:8d}       |"
    format_number_float = "| {:28} | {:14.5f} |"

    print(format_number_string.format('Property', 'Stat'))

    print(format_number_string.format('DOCS',''))
    print(format_number_digit.format('All documents', totnr))
    print(format_number_digit.format('Abstract documents', abstractnr))
    print(format_number_digit.format('Full documents', fullnr))
    print(format_number_float.format('Hypothetical abstract nr', hypothetical_abstracts_nr))

    print(format_number_string.format('TOKENS',''))
    print(format_number_digit.format('All tokens', tot_token))
    print(format_number_digit.format('Abstract doc tokens', abstract_token))
    print(format_number_digit.format('Full doc tokens', full_token))
    print(format_number_float.format('Average tokens per abstract', average_abstract_token))

    print(format_number_string.format('RatioNL',''))
    print(format_number_float.format('RatioNL', rationl))
    print(format_number_digit.format('Number of NL mentions', nl_mention_nr))
    print(format_number_digit.format('Number of All mentions', tot_mention_nr))

    print(format_number_string.format('RatioAbstractFull',''))
    print(format_number_float.format('RatioAbstractFull', ratio_abstract_full))
    print(format_number_float.format('Abstract tokens(NL / All)', nl_norm_abstract))
    print(format_number_float.
          format('Full tokens(NL / All)', nl_norm_full))

    print()


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
    bstrap_path = [path to the directory containing the bootstrapping]
    """

    config_checkdb_help = """
    To Check for integrity of html with ann.json files (offsets, existing ids).
    Only check for integrity and then exit.
    """

    config_stats_demo_help="""
    Generate a graph for several NL Definitions.
    """

    config_iteration_bool_help="""
    Bootstrapping module running on default folder: [resources/bootstrapping] .
    Otherwise the folder from the config files is used under "bstrap_path".
    With Iteration Nr starting automatically. can be specified via -i or --iteration-number.
    """

    config_iteration_number_help= "Manual assignment of number of Iteration"

    parser = argparse.ArgumentParser(description='A simple demo of using the nala pipeline')
    parser.add_argument('-c', '--config', type=argparse.FileType('r'), help=config_ini_help, default='config.ini')
    parser.add_argument('-db', '--check-db', action='store_true', help=config_checkdb_help)
    parser.add_argument('--stats-demo', action='store_true', help=config_stats_demo_help)
    parser.add_argument('-qnl', '--quick-nl', action='store_true', help='Quick Run w/o tokenizer, labeler and crfsuite')
    parser.add_argument('--iteration', action='store_true', help=config_iteration_bool_help)
    parser.add_argument('-i', '--iteration-number', type=int, help=config_iteration_number_help)
    parser.add_argument('--reviewed', action='store_true', help="Manual Annotation was done and saved into reviewed folder")
    args = parser.parse_args()

    config = configparser.ConfigParser()
    try:
        config.read_file(args.config)

        html_path = config.get('paths', 'html_path')
        ann_path = config.get('paths', 'ann_path')
        crf_path = config.get('paths', 'crf_path')
        bstrap_path = config.get('paths', 'bstrap_path')

        # if only to check db then do that...
        if args.check_db:
            dbcheck.main(html_path=html_path, ann_path=ann_path)
            exit()

        if args.iteration:
            if bstrap_path:
                bootstrapping_path = bstrap_path
            else:
                bootstrapping_path = 'resources/bootstrapping'
            bootstrapping_path = os.path.abspath(bootstrapping_path)

            if args.iteration_number:
                iteration = Iteration(folder=bootstrapping_path, iteration_nr=args.iteration_number,
                                      crfsuite_path=crf_path)
            else:
                iteration = Iteration(folder=bootstrapping_path, crfsuite_path=crf_path)

            if args.reviewed:
                iteration.after_annotation()
            else:
                iteration.before_annotation()

            exit()

        dataset = HTMLReader(html_path).read()

        if not args.quick_nl:
            NLTKSplitter().split(dataset)
            NLTKTokenizer().tokenize(dataset)

        AnnJsonAnnotationReader(ann_path, delete_incomplete_docs=False).annotate(dataset)

        if len(dataset.documents) == 0:
            raise IndexError('There are no documents in the dataset. Please check file paths.')

        # ttformat = TagTogFormat(to_save_to="demo/output/", dataset=dataset, who="user:verspoor")
        # ttformat.export_html()
        # ttformat.export_ann_json()
        # TmVarRegexCombinedSelector().define(dataset)
        # exit()

        if args.stats_demo:
            # extra_methods = 2
            # start_min_length = 18
            # end_min_length = 36
            # start_counter = start_min_length - extra_methods

            stats = StatsWriter('csvfile.csv', 'graphfile', init_counter=0)

            # exclusive
            ExclusiveNLDefiner().define(dataset)
            data_stats = dataset.stats()

            # data_stats2 = data_stats.copy()
            # del data_stats2['abstract_nl_mention_array']
            # del data_stats2['nl_mention_array']
            # del data_stats2['full_nl_mention_array']
            # print(json.dumps(data_stats2, indent=4))
            # print(dataset)

            print_stats(data_stats)
            stats.addrow(data_stats, 'Exclusive')
            dataset.clean_nl_definitions()

            # inclusive 18
            InclusiveNLDefiner().define(dataset)
            data_stats = dataset.stats()
            print_stats(data_stats)
            stats.addrow(data_stats, 'Inclusive_18')
            dataset.clean_nl_definitions()

            # inclusive 28
            InclusiveNLDefiner(min_length=28).define(dataset)
            data_stats = dataset.stats()
            print_stats(data_stats)
            stats.addrow(data_stats, 'Inclusive_20')
            dataset.clean_nl_definitions()

            exit()

            # tmvar nl
            # TmVarNLDefiner().define(dataset)
            # stats.addrow(dataset.stats(), 'tmVarComplete')
            # dataset.clean_nl_definitions()

            # inclusive
            # for i in range(start_min_length, end_min_length + 1):
            #     print('run', i)
            #     InclusiveNLDefiner(min_length=i).define(dataset)
            #     inclusivestats = dataset.stats()
            #     inclusivementions = inclusivestats['nl_mention_array']
            #     intersectionset = [ x for x in inclusivementions if x not in tmvarmentions]
            #     print(intersectionset)
            #     stats.addrow(dataset.stats(), 'Inclusive_' + str(i))
            #     dataset.clean_nl_definitions()

            # finally generation of graph itself
            stats.makegraph()

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
