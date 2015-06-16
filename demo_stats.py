from configparser import ConfigParser

from nala.utils.readers import HTMLReader
from nala.utils.annotation_readers import AnnJsonAnnotationReader
from nala.preprocessing.definers import TmVarRegexNLDefiner
from nala.preprocessing.definers import InclusiveNLDefiner
from nala.preprocessing.definers import ExclusiveNLDefiner
from nala.preprocessing.definers import TmVarNLDefiner
from nala.utils.writers import StatsWriter


if __name__ == "__main__":
    # Please define a config.ini file located at nala/config.ini,
    # where you specify the path to the training data and crfsuite on your system.
    #
    # Note: This script and the config file will not be included in the final distribution of nala.
    #       We might however end up producing a demo script similar to this one.
    #       They are for internal purposes only.
    #
    # EXAMPLE: config.ini
    #
    # [paths]
    # html_path = C:\Users\Aleksandar\Desktop\Json and Html\IDP4_plain_html\pool
    # ann_path = C:\Users\Aleksandar\Desktop\Json and Html\IDP4_members_json\pool\abojchevski
    # crf_path = F:\Projects\crfsuite

    try:
        config = ConfigParser()
        config.read('nala/config.ini')

        html_path = config['paths']['html_path']
        ann_path = config['paths']['ann_path']
        crf_path = config['paths']['crf_path']

        dataset = HTMLReader(html_path).read()

        # NLTKSplitter().split(dataset)
        # NLTKTokenizer().tokenize(dataset)

        # prepare
        AnnJsonAnnotationReader(ann_path).annotate(dataset)

        extra_methods = 3
        start_min_length = 18
        end_min_length = 36
        start_counter = start_min_length - extra_methods

        stats = StatsWriter('csvfile.csv', 'graphfile', init_counter=start_counter)

        # tmvar regex
        TmVarRegexNLDefiner().define(dataset)
        tmvarstats = dataset.stats()
        tmvarmentions = tmvarstats['nl_mention_array']
        stats.addrow(tmvarstats, 'tmregex')
        dataset.cleannldefinitions()

        # exclusive
        ExclusiveNLDefiner().define(dataset)
        stats.addrow(dataset.stats(), 'exclusive')
        dataset.cleannldefinitions()

        # tmvar nl
        TmVarNLDefiner().define(dataset)
        stats.addrow(dataset.stats(), 'tmnl')
        dataset.cleannldefinitions()

        # inclusive
        for i in range(start_min_length, end_min_length + 1):
            print('run', i)
            InclusiveNLDefiner(min_length=i).define(dataset)
            inclusivestats = dataset.stats()
            inclusivementions = inclusivestats['nl_mention_array']
            intersectionset = [ x for x in inclusivementions if x not in tmvarmentions]
            print(intersectionset)
            stats.addrow(dataset.stats(), 'inclusive_' + str(i))
            dataset.cleannldefinitions()


        stats.makegraph()

        # print('\n'.join([ann.text for ann in dataset.annotations() if ann.is_nl]))  # print the NL ones

        # SimpleLabeler().label(dataset)
        # SimpleFeatureGenerator().generate(dataset)
        #
        # crf = CRFSuite(crf_path)
        # crf.create_input_file(dataset, 'train')
        # crf.train()
        # crf.create_input_file(dataset, 'test')
        # crf.test()
    except KeyError:
        print('Please define a config.ini file as described above')
