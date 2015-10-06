import argparse
import os
from nala.utils.readers import TextFilesReader
from nala.utils.readers import StringReader
from nala.utils.writers import ConsoleWriter
from nala.structures.dataset_pipelines import PrepareDatasetPipeline
from nala.learning.crfsuite import CRFSuite
import pkg_resources
from nala.learning.taggers import CRFSuiteMutationTagger
from nala.utils import MUT_CLASS_ID


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

    PrepareDatasetPipeline().execute(dataset)

    # get the predictions
    crf = CRFSuite(args.crf_suite_dir)
    tagger = CRFSuiteMutationTagger([MUT_CLASS_ID], crf, pkg_resources.resource_filename('nala.data', 'default_model'))
    tagger.tag(dataset)

    PostProcessing().process(dataset)
    ConsoleWriter(args.color).write(dataset)
