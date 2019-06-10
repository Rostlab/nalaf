import warnings

import pycrfsuite

from nalaf.structures.data import Label


class PyCRFSuite:

    def __init__(self, model_file=None):
        self.model_file = model_file

        if self.model_file is None:
            self.tagger = None
        else:
            self.tagger = pycrfsuite.Tagger()
            self.tagger.open(self.model_file)


    def close(self):
        if self.tagger is not None:
            self.tagger.close()


    def __del__(self):
        self.close()


    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


    def annotate(self, corpus, class_id):
        """
        :type corpus: nalaf.structures.data.Dataset
        :type class_id: str ~ to annotate with
        """

        for sentence in corpus.sentences():
            labels = self.tagger.tag(pycrfsuite.ItemSequence(token.features for token in sentence))

            for token_index in range(len(sentence)):
                label = labels[token_index]
                try:
                    sentence[token_index].predicted_labels = [Label(label, self.tagger.marginal(label, token_index))]
                except Exception as e:
                    raise Exception("Exception when assining the predicted labels; likely a Multi-Thread problem", e)

        corpus.form_predicted_annotations(class_id)


    @staticmethod
    def train(data, model_file, params=None):
        """
        :type data: nalaf.structures.data.Dataset
        :type model_file: str ~ filename (from local file system) to save trained model to. If None, no model is saved.
        """

        trainer = pycrfsuite.Trainer()

        try:
            if params is not None:
                trainer.set_params(params)

            for sentence in data.sentences():
                trainer.append(pycrfsuite.ItemSequence([token.features for token in sentence]),
                               [token.original_labels[0].value for token in sentence])

            # The CRFSuite library handles the "pickling" of the file; saves the model here
            trainer.train(model_file)

        finally:
            trainer.clear()


    @staticmethod
    def tag(data, model_file, class_id):
        warnings.warn('Use non-static `annotate` instead', DeprecationWarning)

        """
        :type data: nalaf.structures.data.Dataset
        :type model_file: str
        """

        tagger = pycrfsuite.Tagger()

        try:
            tagger.open(model_file)

            for sentence in data.sentences():
                labels = tagger.tag(pycrfsuite.ItemSequence(token.features for token in sentence))

                for token_index in range(len(sentence)):
                    label = labels[token_index]
                    sentence[token_index].predicted_labels = [Label(label, tagger.marginal(label, token_index))]

            data.form_predicted_annotations(class_id)

        finally:
            tagger.close()
