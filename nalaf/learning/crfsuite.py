import os
import sys
import warnings

import pycrfsuite

from nalaf.structures.data import Label
from nalaf.learning.taggers import Tagger


class PyCRFSuite:

    def __init__(self, model_file=None):
        self.model_file = model_file

        if self.model_file is None:
            self.tagger = None
        else:
            self.tagger = pycrfsuite.Tagger()
            self.tagger.open(self.model_file)


    def annotate(self, corpus, class_id):
        """
        :type corpus: nalaf.structures.data.Dataset
        :type class_id: str ~ to annotate with
        """

        for sentence in corpus.sentences():
            labels = self.tagger.tag(pycrfsuite.ItemSequence(token.features for token in sentence))

            for token_index in range(len(sentence)):
                label = labels[token_index]
                sentence[token_index].predicted_labels = [Label(label, self.tagger.marginal(label, token_index))]

        corpus.form_predicted_annotations(class_id)


    @staticmethod
    def train(data, model_file, params=None):
        """
        :type data: nalaf.structures.data.Dataset
        :type model_file: str ~ filename (from local file system) to save trained model to. If None, no model is saved.
        """

        trainer = pycrfsuite.Trainer()
        if params is not None:
            trainer.set_params(params)

        for sentence in data.sentences():
            trainer.append(pycrfsuite.ItemSequence([token.features for token in sentence]),
                           [token.original_labels[0].value for token in sentence])

        # The CRFSuite library handles the "pickling" of the file; saves the model here
        trainer.train(model_file)


    @staticmethod
    def tag(data, model_file, class_id):
        warnings.warn('Use non-static `annotate` instead', DeprecationWarning)

        """
        :type data: nalaf.structures.data.Dataset
        :type model_file: str
        """

        tagger = pycrfsuite.Tagger()
        tagger.open(model_file)

        for sentence in data.sentences():
            labels = tagger.tag(pycrfsuite.ItemSequence(token.features for token in sentence))

            for token_index in range(len(sentence)):
                label = labels[token_index]
                sentence[token_index].predicted_labels = [Label(label, tagger.marginal(label, token_index))]

        data.form_predicted_annotations(class_id)


class CRFSuite:
    """
    Basic class for interaction with CRFSuite
    """

    def __init__(self, directory, minify=False):
        warnings.warn('Deprecated. Please use PyCRFSuite instead', DeprecationWarning)

        self.directory = os.path.abspath(directory)
        """the directory where the CRFSuite executable is located"""
        self.model_filename = 'example_entity_model'
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

        :type dataset: nalaf.structures.data.Dataset
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


    def read_predictions(self, dataset, class_id, prediction_file='output.txt'):
        """
        :type dataset: nalaf.structures.data.Dataset

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
            * crf.test(options='-m example_entity_model -i test > output.txt')
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


class CRFSuiteTagger(Tagger):
    """
    Performs tagging with a binary model using CRFSuite

    :type crf_suite: nalaf.learning.crfsuite.CRFSuite
    """

    def __init__(self, predicts_classes, crf_suite, model_file='example_entity_model'):
        warnings.warn('Use PyCRFSuite', DeprecationWarning)

        super().__init__(predicts_classes)
        self.crf_suite = crf_suite
        """an instance of CRFSuite used to actually generate predictions"""
        self.model_file = model_file
        """path to the binary model used for generating predictions"""

    def tag(self, dataset):
        """
        :type dataset: nalaf.structures.data.Dataset
        """
        self.crf_suite.create_input_file(dataset, 'predict')
        self.crf_suite.tag('-m {} -i predict > output.txt'.format(self.model_file))
        self.crf_suite.read_predictions(dataset)
