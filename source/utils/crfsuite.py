import os


class CRFSuite():
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

        :type dataset: structures.data.Dataset
        :param mode: one of the following 'train' or 'test' or 'predict'
        :type mode: str
        """
        with open('%s/%s' % (self.directory, mode), 'w', encoding='utf-8') as file:

            for sentence in dataset.sentences():
                for token in sentence:
                    features = '\t'.join(['%s=%s' % (key, value.replace(':', '_COLON_')) if type(value) is str
                                          else '%s:%f' % (key, value)
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
        os.system('crfsuite learn %s -m %s train' % (options, self.model_filename))

    def test(self, options=''):
        """
        Test a CRF model with the latest model and test file.
        """
        os.chdir(self.directory)
        os.system('crfsuite tag %s -qt -m %s test' % (options, self.model_filename))