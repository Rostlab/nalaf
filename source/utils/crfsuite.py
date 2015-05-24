import os


class CRFSuite():
    def __init__(self, directory):
        self.directory = directory
        self.model_filename = 'default_model'

    def create_input_file(self, dataset, mode):
        with open('%s/%s' % (self.directory, mode), 'w', encoding='utf-8') as file:

            for sentence in dataset.sentences():
                for token in sentence:
                    features = '\t'.join(['%s=%s' % (key, value) if type(value) is str
                                          else '%s:%f' % (key, value)
                                          for key, value in token.features.items()])

                    if mode in ('train', 'test'):
                        file.write('%s\t%s\n' % (token.original_labels[0], features))
                    else:  # mode=predict
                        file.write('%s\n' % features)
                file.write('\n')

    def train(self):
        os.chdir(self.directory)
        os.system('crfsuite learn -m %s train' % self.model_filename)

    def test(self):
        os.chdir(self.directory)
        os.system('crfsuite tag -qt -m %s test' % self.model_filename)