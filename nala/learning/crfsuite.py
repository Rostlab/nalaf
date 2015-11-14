import os
import sys
from nala.structures.data import Label
from nala.utils import MUT_CLASS_ID

class CRFSuite:
    """
    Basic class for interaction with CRFSuite
    """
    #NOTE: Make the class a bit more generic or replace with an existing package such as python-crfsuite (as for the binding)

    def __init__(self, directory, minify=False):
        self.directory = os.path.abspath(directory)
        """the directory where the CRFSuite executable is located"""
        self.model_filename = 'default_model'
        """name to be used for saving the model"""
        if sys.platform.startswith('linux'):
            self.crf_suite_call = './crfsuite'
        else:
            self.crf_suite_call = 'crfsuite'
        self.minify = minify
        """controls whether to replace feature names with an index in order to minimize input file length"""

    def create_input_file(self, dataset, mode):
        """
        Creates the input files for training, testing or prediction in the appropriate format required by CRFSuite.
        Saves the files in the same directory where the executable is located.

        :type dataset: nala.structures.data.Dataset
        :param mode: one of the following 'train' or 'test' or 'predict'
        :type mode: str
        """
        if self.minify:
            key_map = {key: index for index, key in
                       enumerate(set(key for token in dataset.tokens() for key in token.features.keys()))}
            key_string = lambda key: key_map[key]
        else:
            key_string = lambda key: key

        with open(os.path.join(self.directory, mode), 'w', encoding='utf-8') as file:
            for sentence in dataset.sentences():
                for token in sentence:
                    features = '\t'.join(['{}:{}'.format(key_string(key), value)
                                          if type(value) is float
                                          else '{}={}'.format(key_string(key), str(value).replace(':', '_COLON_'))
                                          for key, value in token.features.items()])

                    if mode in ('train', 'test'):
                        label = token.original_labels[0].value
                    else:
                        label = '?'
                    file.write('{}\t{}\n'.format(label, features))
                file.write('\n')

    def learn(self, options=''):
        """
        Train and save a CRF model with the latest train file.
        """
        os.chdir(self.directory)
        if options:
            os.system('{} learn {}'.format(self.crf_suite_call, options))
        else:
            os.system('{} learn -m {} train'.format(self.crf_suite_call, self.model_filename))

    def tag(self, options=''):
        """
        Test a CRF model with the latest model and test file.
        """
        os.chdir(self.directory)
        if options:
            os.system('{} tag {}'.format(self.crf_suite_call, options))
        else:
            os.system('{} tag -qt -m {} test'.format(self.crf_suite_call, self.model_filename))

    def read_predictions(self, dataset, prediction_file='output.txt', class_id = MUT_CLASS_ID):
        """
        :type dataset: nala.structures.data.Dataset

        Reads in the predictions made by our model for each token and stores them into token.predicted_label[]

        Requires a dataset object and the output prediction file.

        The default output prediction file is 'output.txt'. The format is:
            * [predicted label]:[marginal probability]
            * in new line for each token
            * followed by a blank line for the end of the sentence

        IMPORTANT NOTE:
        Assumes a call to the test() function was made previously with the 'i' option included.
        Furthermore, it assumes we are calling it with the same dataset object used to create the test file.

        For example first we would call:
            * crf.create_input_file(dataset=test, mode='test')
            * crf.test(options='-m default_model -i test > output.txt')
        Then we would call:
            * crf.read_predictions(dataset=test)
        """

        os.chdir(self.directory)
        with open(prediction_file) as file:
            for sentence in dataset.sentences():
                for token in sentence:
                    label, probability = file.readline().split(':')
                    token.predicted_labels = [Label(label, float(probability))]

                file.readline()  # skip the empty line signifying new sentence

        # call form_predicted_annotations() to populate the mention level predictions
        dataset.form_predicted_annotations(class_id)
