import os
import sys
import subprocess

from nala.structures.data import Label
from nala.utils import MUT_CLASS_ID
from random import random, seed


class SVMLightTreeKernels:
    """
    Base class for interaction with Alessandro Moschitti's Tree Kernels in
    SVM Light
    """
    def __init__(self, directory='resources/svmlight/', model='default_model', use_tree_kernel=True):
        self.directory = directory
        """the directory where the executables svm_classify and svm_learn are
        located"""
        self.model = model
        """the model to read from / write to"""
        self.use_tree_kernel = use_tree_kernel
        """whether to use tree kernels or not"""
        if sys.platform.startswith('linux'):
            self.svm_learn_call = os.path.join(self.directory, 'svm_learn')
            self.svm_classify_call = os.path.join(self.directory, 'svm_classify')
        else:
            self.svm_learn_call = os.path.join(self.directory, 'svm_learn.exe')
            self.svm_classify_call = os.path.join(self.directory, 'svm_classify.exe')

    def create_input_file(self, dataset, mode, features, undersampling=0.4, minority_class=-1, file=None):
        string = ''
        if mode=='train':
            for edge in dataset.edges():
                if edge.target==minority_class:
                    prob = random()
                    if prob<undersampling:
                        string += str(edge.target)
                        if self.use_tree_kernel:
                            string += ' |BT| '
                            string += edge.part.sentence_parse_trees[edge.sentence_id]
                            string += ' |ET|'
                        values = set(features.values())
                        for key in sorted(edge.features.keys()):
                            if key in values:
                                value = edge.features[key]
                                string += ' '+str(key)+':'+str(value)
                        string += '\n'
                else:
                    string += str(edge.target)
                    if self.use_tree_kernel:
                        string += ' |BT| '
                        string += edge.part.sentence_parse_trees[edge.sentence_id]
                        string += ' |ET|'
                    values = set(features.values())
                    for key in sorted(edge.features.keys()):
                        if key in values:
                            value = edge.features[key]
                            string += ' '+str(key)+':'+str(value)
                    string += '\n'
        elif mode=='test':
            for edge in dataset.edges():
                string += str(edge.target)
                if self.use_tree_kernel:
                    string += ' |BT| '
                    string += edge.part.sentence_parse_trees[edge.sentence_id]
                    string += ' |ET|'
                values = set(features.values())
                for key in sorted(edge.features.keys()):
                    if key in values:
                        value = edge.features[key]
                        string += ' '+str(key)+':'+str(value)
                string += '\n'
        elif mode=='predict':
            for edge in dataset.edges():
                string += '?'
                if self.use_tree_kernel:
                    string += ' |BT| '
                    string += edge.part.sentence_parse_trees[edge.sentence_id]
                    string += ' |ET|'
                for key in sorted(edge.features.keys()):
                    if key in features.values():
                        value = edge.features[key]
                        string += ' '+str(key)+':'+str(value)
                string += '\n'
        if file is None:
            file = os.path.join(self.directory, mode)
        with open(file, 'w', encoding='utf-8') as f:
            f.write(string)

    def learn(self, file=None, c=0.5):
        if file is None:
            file = os.path.join(self.directory, 'train')
        if self.use_tree_kernel:
            subprocess.call([
    					self.svm_learn_call,
                        '-v', '0',
    					'-t', '5',
    					'-T', '1',
    					'-W', 'S',
    					'-V', 'S',
    					'-C', '+',
                        '-c', str(c),
    					file,
    					os.path.join(self.directory, self.model),
    					])
        else:
            subprocess.call([
                        self.svm_learn_call,
                        '-c', str(c),
                        '-v', '0',
                        file,
                        os.path.join(self.directory, self.model),
            ])

    def tag(self, file=None, mode='predict', output=None):
        if file is None:
            file = os.path.join(self.directory, mode)
        if output is None:
            output = os.path.join(self.directory, 'predictions')
        subprocess.call([
                        self.svm_classify_call,
                        '-v', '0',
                        file,
                        os.path.join(self.directory, self.model),
                        output
                        ])

    def read_predictions(self, dataset, predictions=None):
        if predictions is None:
            predictions = os.path.join(self.directory, 'predictions')
        values = []
        with open(predictions) as file:
            for line in file:
                if float(line.strip())>-0.1:
                    values.append(1)
                else:
                    values.append(-1)
        for index, edge in enumerate(dataset.edges()):
            edge.target = values[index]
        dataset.form_predicted_relations()
