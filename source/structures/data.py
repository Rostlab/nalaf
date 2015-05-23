class Label():
    def __init__(self, value, confidence=None):
        self.value = value
        self.confidence = confidence


class Token():
    def __init__(self, word):
        self.word = word
        self.original_labels = None
        self.predicted_labels = None
        self.features = {}

    def __repr__(self):
        return self.word


class Annotation():
    def __init__(self, class_id, offset, text):
        self.class_id = class_id
        self.offset = offset
        self.text = text


class Part():
    def __init__(self, text):
        self.text = text
        self.sentences = [[]]
        self.annotations = []

    def __iter__(self):
        return iter(self.sentences)


class Document():
    def __init__(self):
        self.parts = {}

    def __iter__(self):
        for part_id, part in self.parts.items():
            yield part


class Dataset():
    def __init__(self):
        self.documents = {}

    def __iter__(self):
        for doc_id, document in self.documents.items():
            yield document

    def parts(self):
        for document in self:
            for part in document:
                yield part

    def sentences(self):
        for part in self.parts():
            for sentence in part.sentences:
                yield sentence

    def tokens(self):
        for sentence in self.sentences():
            for token in sentence:
                yield token