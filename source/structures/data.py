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


class Part():
    def __init__(self, text):
        self.text = text
        self.sentences = [[]]

    def __iter__(self):
        return iter(self.sentences)


class Document():
    def __init__(self, id):
        self.id = id
        self.parts = []
        self.source = None

    def __iter__(self):
        return iter(self.parts)