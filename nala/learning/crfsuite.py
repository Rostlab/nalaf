import os
import sys
from nala.structures.data import Label

class CRFSuite:
    """
    Basic class for interaction with CRFSuite

    TODO: Make the class a bit more generic or replace with an existing package such as python-crfsuite
    """

    def __init__(self, directory):
        self.directory = os.path.abspath(directory)
        """the directory where the CRFSuite executable is located"""
        self.model_filename = 'default_model'
        """name to be used for saving the model"""
        if sys.platform.startswith('linux'):
            self.crf_suite_call = './crfsuite'
        else:
            self.crf_suite_call = 'crfsuite'

    def create_input_file(self, dataset, mode):
        """
        Creates the input files for training, testing or prediction in the appropriate format required by CRFSuite.
        Saves the files in the same directory where the executable is located.

        :type dataset: nala.structures.data.Dataset
        :param mode: one of the following 'train' or 'test' or 'predict'
        :type mode: str
        """
        with open('%s/%s' % (self.directory, mode), 'w', encoding='utf-8') as file:

            for sentence in dataset.sentences():
                for token in sentence:
                    features = '\t'.join(['{}={}'.format(key, str(value).replace(':', '_COLON_')) if 'embedding' not in key
                                          else '{}:{}'.format(key, value)
                                          for key, value in token.features.items()])

                    if mode in ('train', 'test'):
                        file.write('%s\t%s\n' % (token.original_labels[0], features))
                    else:  # mode=predict
                        file.write('%s\n' % features)
                file.write('\n')

    def train(self, options=''):
        """
        Train and save a CRF model with the latest train file.
        """
        os.chdir(self.directory)
        if options:
            os.system('{} learn {}'.format(self.crf_suite_call, options))
        else:
            os.system('{} learn -m {} train'.format(self.crf_suite_call, self.model_filename))

    def test(self, options=''):
        """
        Test a CRF model with the latest model and test file.
        """
        os.chdir(self.directory)
        if options:
            os.system('{} tag {}'.format(self.crf_suite_call, options))
        else:
            os.system('{} tag -qt -m {} test'.format(self.crf_suite_call, self.model_filename))

    def read_predictions(self, dataset, prediction_file='output.txt'):
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
        dataset.form_predicted_annotations()
