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

    def key_value_parts(self):
        """yields iterator for partids"""
        for part_id, part in self.parts.items():
            yield part_id, part


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

    def partids_with_parts(self):
        """ helper function that yields part id with part"""
        for document in self:
            for part_id, part in document.key_value_parts():
                yield part_id, part

    def annotations_with_partids(self):
        """ helper function that return annotation object with part id
        to be able to find out abstract or full document """
        for part_id, part in self.partids_with_parts():
            for annotation in part.annotations:
                yield part_id, annotation

    def stats(self):
        """
        Calculates stats on the dataset. Like amount of nl mentions, ....
        """
        import re

        # main values
        nl_mentions = []  # array of nl mentions each of the the whole ann.text saved
        nl_nr = 0  # total nr of nl mentions
        nl_token_nr = 0  # total nr of nl tokens
        mentions_nr = 0  # total nr of all mentions (including st mentions)
        mentions_token_nr = 0  # total nr of all tokens of all mentions (inc st mentions)
        total_token_abstract = 0
        total_token_full = 0

        # abstract nr
        abstract_mentions_nr = 0
        abstract_token_nr = 0
        abstract_nl_mentions = []

        # full document nr
        full_document_mentions_nr = 0
        full_document_token_nr = 0
        full_nl_mentions = []

        # is abstract var
        is_abstract = True

        # precompile abstract match
        regex_abstract_id = re.compile(r'^s[12][shp]')

        for part_id, ann in self.annotations_with_partids():
            # abstract?
            if re.match(regex_abstract_id, part_id):
                is_abstract = True
            else:
                is_abstract = False

            if ann.class_id == 'e_2':
                # preprocessing
                token_nr = len(ann.text.split(" "))
                mentions_nr += 1
                mentions_token_nr += token_nr

                if ann.is_nl:
                    # total nr increase
                    nl_nr += 1
                    nl_token_nr += token_nr

                    # abstract nr of tokens increase
                    if is_abstract:
                        abstract_mentions_nr += 1
                        abstract_token_nr += token_nr
                        abstract_nl_mentions.append(ann.text)
                    else:
                        # full document nr of tokens increase
                        full_document_mentions_nr += 1
                        full_document_token_nr += token_nr
                        full_nl_mentions.append(ann.text)

                    # nl text mention add to []
                    nl_mentions.append(ann.text)

        # post-processing for abstract vs full document tokens
        for part_id, part in self.partids_with_parts():
            if re.match(regex_abstract_id, part_id):
                # OPTIONAL use nltk or different tokenizer
                total_token_abstract += len(part.text.split(" "))
            else:
                total_token_full += len(part.text.split(" "))


        # report_array = []
        #
        # report_array.extend([nl_nr, mentions_nr, nl_token_nr, mentions_token_nr, abstract_mentions_nr, abstract_token_nr])
        # report_array.extend([full_document_mentions_nr, full_document_token_nr, total_token_abstract, total_token_full])
        # report_array.extend(nl_mentions)

        report_dict = {
            'nl_mention_nr': nl_nr,
            'tot_mention_nr': mentions_nr,
            'nl_token_nr': nl_token_nr,
            'tot_token_nr': mentions_token_nr,
            'abstract_nl_mention_nr': abstract_mentions_nr,
            'abstract_nl_token_nr': abstract_token_nr,
            'abstract_tot_token_nr': total_token_abstract,
            'full_nl_mention_nr': full_document_mentions_nr,
            'full_nl_token_nr': full_document_token_nr,
            'full_tot_token_nr': total_token_full,
            'nl_mention_array': sorted(nl_mentions),
            'abstract_nl_mention_array': sorted(abstract_nl_mentions),
            'full_nl_mention_array': sorted(full_nl_mentions)
        }

        # if total_token_abstract > 0 and total_token_full > 0:
        #     # ratio calc (token nl)/(token total) from abstract
        #     token_nl_vs_tot_abstract = abstract_token_nr / float(total_token_abstract)
        #     # ratio calc (token nl)/(token total) from abstract
        #     token_nl_vs_tot_full = full_document_token_nr / float(total_token_full)
        #     # ratio-(token nl mentions/total tokens) in abstract ratio full documents
        #     if token_nl_vs_tot_full > 0:
        #         ratio_nl_men = token_nl_vs_tot_abstract / token_nl_vs_tot_full
        #         report_array.append(ratio_nl_men)
        #     else:
        #         report_array.append("None")
        # else:
        #     report_array.append("None")
        # print("nl mentions", nl_nr, "total mentions", mentions_nr, nl_nr/mentions_nr)

        return report_dict
        # FIXME return object as dictionary
        # TODO (1) ratio-(token nl mentions/total tokens) in abstract ratio full documents @graph
        # TODO (3) tmvar regex intersection mutations check --> export list of mentions @graph @export
        # TODO (2) ratio-(nl/total) @graph
        # TODO (4) nl mentions total vs min lettre parameter @graph @parameters
        # TODO (5) parametrizable min length (12..36) @parameters
