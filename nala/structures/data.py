class Label:
    """
    Represents the label associated with each Token.
    """

    def __init__(self, value, confidence=None):
        self.value = value
        """string value of the label"""
        self.confidence = confidence
        """probability of being correct if the label is predicted"""

    def __repr__(self):
        return self.value


class Token:
    """
    Represent a token - the smallest unit on which we perform operations.
    Usually one token represent one word from the document.
    """

    def __init__(self, word):
        self.word = word
        """string value of the token, usually a single word"""
        self.original_labels = None
        """the original labels for the token as assigned by some implementation of Labeler"""
        self.predicted_labels = None
        """the predicted labels for the token as assigned by some learning algorightm"""
        self.features = {}
        """
        a dictionary of features for the token
        each feature is represented as a key value pair:
        * [string], [string] pair denotes the feature "[string]=[string]"
        * [string], [float] pair denotes the feature "[string]:[float] where the [float] is a weight"
        """

    def __repr__(self):
        """
        print calls to the class Token will print out the string contents of the word
        """
        return self.word


class Annotation:
    """
    Represent a single annotation, that is denotes a span of text which represents some entitity.
    """

    def __init__(self, class_id, offset, text):
        self.class_id = class_id
        """the id of the class or entity that is annotated"""
        self.offset = offset
        """the offset marking the beginning of the annotation in regards to the Part this annotation is attached to."""
        self.text = text
        """the text span of the annotation"""
        self.is_nl = False
        """boolean indicator if the annotation is a natural language (NL) mention."""


class Part:
    """
    Represent chunks of text grouped in the document that for some reason belong together.
    Each part hold a reference to the annotations for that chunk of text.
    """

    def __init__(self, text):
        self.text = text
        """the original raw text that the part is consisted of"""
        self.sentences = [[]]
        """
        a list sentences where each sentence is a list of tokens
        derived from text by calling Splitter and Tokenizer
        """
        self.annotations = []
        """the annotations of the chunk of text as populated by a call to Annotator"""

    def __iter__(self):
        """
        when iterating through the part iterate through each sentence
        """
        return iter(self.sentences)


class Document:
    """
    Class representing a single document, for example an article from PubMed.
    """

    def __init__(self):
        self.parts = {}
        """
        parts the document consists of, encoded as a dictionary
        where the key (string) is the id of the part
        and the value is an instance of Part
        """

    def __iter__(self):
        """
        when iterating through the document iterate through each part
        """
        for part_id, part in self.parts.items():
            yield part


class Dataset:
    """
    Class representing a group of documents.
    Instances of this class are the main object that gets passed around and modified by different modules.
    """

    def __init__(self):
        self.documents = {}
        """
        documents the dataset consists of, encoded as a dictionary
        where the key (string) is the id of the document, for example PubMed id
        and the value is an instance of Document
        """

    def __len__(self):
        """
        the length (size) of a dataset equals to the number of documents it has
        """
        return len(self.documents)

    def __iter__(self):
        """
        when iterating through the dataset iterate through each document
        """
        for doc_id, document in self.documents.items():
            yield document

    def parts(self):
        """
        helper functions that iterates through all parts
        that is each part of each document in the dataset
        """
        for document in self:
            for part in document:
                yield part

    def annotations(self):
        """
        helper functions that iterates through all parts
        that is each part of each document in the dataset
        """
        for part in self.parts():
            for annotation in part.annotations:
                yield annotation

    def sentences(self):
        """
        helper functions that iterates through all sentences
        that is each sentence of each part of each document in the dataset
        """
        for part in self.parts():
            for sentence in part.sentences:
                yield sentence

    def tokens(self):
        """
        helper functions that iterates through all tokens
        that is each token of each sentence of each part of each document in the dataset
        """
        for sentence in self.sentences():
            for token in sentence:
                yield token

    def stats(self):
        """
        Calculates stats on the dataset. Like amount of nl mentions, ....
        """
        # FIXME Add more information (30) --> detailed @bug
