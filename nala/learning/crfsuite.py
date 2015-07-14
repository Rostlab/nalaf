import os
from nala.structures.data import Label

class CRFSuite:
    """
    Basic class for interaction with CRFSuite

    TODO: Make the class a bit more generic or replace with an existing package such as python-crfsuite
    """

    def __init__(self, directory):
        self.directory = directory
        """the directory where the CRFSuite executable is located"""
        self.model_filename = 'default_model'
        """name to be used for saving the model"""

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
            os.system('crfsuite learn {}'.format(options))
        else:
            os.system('crfsuite learn -m {} train'.format(self.model_filename))

    def test(self, options=''):
        """
        Test a CRF model with the latest model and test file.
        """
        os.chdir(self.directory)
        if options:
            os.system('crfsuite tag {}'.format(options))
        else:
            os.system('crfsuite tag -qt -m {} test'.format(self.model_filename))

    def read_predictions(self, dataset, prediction_file='output.txt'):
        # TODO fix docstring
        """
        Reads in the predictions made by our model
        and stores them into the token.predicted_label[]

        :type dataset: nala.structures.data.Dataset
        """

        os.chdir(self.directory)
        with open(prediction_file) as file:
            for sentence in dataset.sentences():
                for token in sentence:
                    label, probablity = file.readline().split(':')
                    token.predicted_labels = [Label(label, float(probablity))]

                file.readline()  # skip the empty line signifying new sentence

