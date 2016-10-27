import os
import sys
import subprocess
from random import random
import tempfile
from nalaf import print_warning, print_verbose, print_debug


class SVMLightTreeKernels:
    """
    Base class for interaction with Alessandro Moschitti's Tree Kernels in SVM Light
    """

    def __init__(self, svmlight_dir_path='', model_path=tempfile.NamedTemporaryFile().name, use_tree_kernel=True):
        self.svmlight_dir_path = svmlight_dir_path
        """
        The directory where the executables svm_classify and svm_learn are located.
        Defaults to the empty string '', which then means that the svmlight executables must be in your binary path
        """

        executables_extension = '' if sys.platform.startswith('linux') or sys.platform.startswith('darwin') else '.exe'
        self.svm_learn_call = os.path.join(self.svmlight_dir_path, ('svm_learn' + executables_extension))
        self.svm_classify_call = os.path.join(self.svmlight_dir_path, ('svm_classify' + executables_extension))

        self.model_path = model_path
        """the model (path) to read from / write to"""

        self.use_tree_kernel = use_tree_kernel
        """whether to use tree kernels or not"""


    def create_input_file(self, dataset, mode, features, undersampling=0.4, minority_class=-1):
        string = ''

        if mode == 'train':
            for edge in dataset.edges():
                if edge.target == minority_class:
                    prob = random()
                    if prob < undersampling:
                        string += str(edge.target)
                        if self.use_tree_kernel:
                            string += ' |BT| '
                            string += edge.part.sentence_parse_trees[edge.sentence_id]
                            string += ' |ET|'
                        values = set(features.values())
                        for key in sorted(edge.features.keys()):
                            if key in values:
                                value = edge.features[key]
                                string += ' ' + str(key) + ':' + str(value)
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
                            string += ' ' + str(key) + ':' + str(value)
                    string += '\n'

        elif mode == 'test':
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
                        string += ' ' + str(key) + ':' + str(value)
                string += '\n'

        elif mode == 'predict':
            for edge in dataset.edges():
                string += '?'
                if self.use_tree_kernel:
                    string += ' |BT| '
                    string += edge.part.sentence_parse_trees[edge.sentence_id]
                    string += ' |ET|'
                for key in sorted(edge.features.keys()):
                    if key in features.values():
                        value = edge.features[key]
                        string += ' ' + str(key) + ':' + str(value)
                string += '\n'

        instancesfile = tempfile.NamedTemporaryFile('w', delete=False)
        print_debug("Instances file: " + instancesfile.name)
        instancesfile.write(string)

        return instancesfile


    def learn(self, instancesfile, c=0.5):

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
                instancesfile.name,
                self.model_path
            ])

        else:
            subprocess.call([
                self.svm_learn_call,
                '-c', str(c),
                '-v', '0',
                instancesfile.name,
                self.model_path
            ])

        return self.model_path


    def tag(self, instancesfile):

        predictionsfile = tempfile.NamedTemporaryFile('r+', delete=False)
        print_debug("Predictions file: " + predictionsfile.name)

        call = [
            self.svm_classify_call,
            '-v', '0',
            instancesfile.name,
            self.model_path,
            predictionsfile.name
        ]
        exitcode = subprocess.call(call)

        if exitcode != 0:
            raise Exception("Error when tagging: " + ' '.join(call))

        return predictionsfile


    def read_predictions(self, dataset, predictionsfile):
        values = []
        with predictionsfile:
            predictionsfile.seek(0)

            for line in predictionsfile:
                if float(line.strip()) > -0.1:
                    values.append(1)
                else:
                    values.append(-1)

            if (len(values) > 1):
                for index, edge in enumerate(dataset.edges()):
                    edge.target = values[index]
            else:
                raise Exception("EMPTY PREDICTIONS FILE -- This may be due to too small dataset or too few of features. Predictions file: " + predictionsfile.name)

        return dataset.form_predicted_relations()
