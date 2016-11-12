from collections import OrderedDict
from itertools import chain
import json
import random
import re
from nalaf.utils.qmath import arithmetic_mean
from nalaf import print_debug, print_verbose
import warnings


class Dataset:
    """
    Class representing a group of documents.
    Instances of this class are the main object that gets passed around and modified by different modules.

    :type documents: dict
    """

    def __init__(self):
        self.documents = OrderedDict()
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

    def __contains__(self, item):
        return item in self.documents


    def parts(self):
        """
        helper functions that iterates through all parts
        that is each part of each document in the dataset

        :rtype: collections.Iterable[Part]
        """
        for document in self:
            for part in document:
                yield part


    def annotations(self):
        """
        helper functions that iterates through all parts
        that is each part of each document in the dataset

        :rtype: collections.Iterable[Entity]
        """
        # TODO
        warnings.warn('annotations actually means entities. This method and related attributes will soon be renamed')

        for part in self.parts():
            for annotation in part.annotations:
                yield annotation


    def predicted_annotations(self):
        """
        helper functions that iterates through all parts
        that is each part of each document in the dataset

        :rtype: collections.Iterable[Entity]
        """
        # TODO
        warnings.warn('annotations actually means entities. This method and related attributes will soon be renamed')

        for part in self.parts():
            for annotation in part.predicted_annotations:
                yield annotation


    def relations(self):
        """
        helper function that iterates through all relations
        :rtype: collections.Iterable[Relation]
        """
        for part in self.parts():
            for rel in part.relations:
                yield rel


    def predicted_relations(self):
        """
        helper function that iterates through all predicted relations
        :rtype: collections.Iterable[Relation]
        """
        for part in self.parts():
            for relation in part.predicted_relations:
                yield relation


    def sentences(self):
        """
        helper functions that iterates through all sentences
        that is each sentence of each part of each document in the dataset

        :rtype: collections.Iterable[list[Token]]
        """
        for part in self.parts():
            for sentence in part.sentences:
                yield sentence


    def tokens(self):
        """
        helper functions that iterates through all tokens
        that is each token of each sentence of each part of each document in the dataset

        :rtype: collections.Iterable[Token]
        """
        for sentence in self.sentences():
            for token in sentence:
                yield token


    def edges(self):
        """
        helper function that iterations through all edges
        that is, each edge of each sentence of each part of each document in the dataset

        :rtype: collections.Iterable[Edge]
        """
        for part in self.parts():
            for edge in part.edges:
                yield edge


    def label_edges(self):
        """
        label each edge with its target - whether it is indeed a relation or not
        """
        for edge in self.edges():
            if edge.is_relation():
                edge.target = 1
            else:
                edge.target = -1


    def purge_false_relationships(self):
        """
        cleans false relationships by validating them
        :return:
        """
        for part in self.parts():
            part.relations[:] = [x for x in part.relations if x.validate_itself(part)]


    def partids_with_parts(self):
        """
        helper function that yields part id with part

        :rtype: collections.Iterable[(str, Part)]
        """
        for document in self:
            for part_id, part in document.key_value_parts():
                yield part_id, part


    def annotations_with_partids(self):
        """
        helper function that return annotation object with part id
        to be able to find out abstract or full document

        :rtype: collections.Iterable[(str, Entity)]
        """
        # TODO
        warnings.warn('annotations actually means entities. This method and related attributes will soon be renamed')

        for part_id, part in self.partids_with_parts():
            for annotation in part.annotations:
                yield part_id, annotation

    def all_annotations_with_ids(self):
        """
        yields pubmedid, partid and ann through whole dataset

        :rtype: collections.Iterable[(str, str, Entity)]
        """
        # TODO
        warnings.warn('annotations actually means entities. This method and related attributes will soon be renamed')

        for pubmedid, doc in self.documents.items():
            for partid, part in doc.key_value_parts():
                for ann in part.annotations:
                    yield pubmedid, partid, ann

    def all_annotations_with_ids_and_is_abstract(self):
        """
        yields pubmedid, partid, is_abstract and ann through whole dataset

        :rtype: collections.Iterable[(str, str, bool, Entity)]
        """
        # TODO
        warnings.warn('annotations actually means entities. This method and related attributes will soon be renamed')

        for pubmedid, doc in self.documents.items():
            for partid, part in doc.key_value_parts():
                for ann in part.annotations:
                    yield pubmedid, partid, part.is_abstract, ann


    def form_predicted_annotations(self, class_id, aggregator_function=arithmetic_mean):
        """
        Populates part.predicted_annotations with a list of Annotation objects
        based on the values of the field predicted_label for each token.

        One annotation is the chunk of the text (e.g. mutation mention)
        whose tokens have labels that are continuously not 'O'
        For example:
        ... O O O A D I S A O O O ...
        ... ----- annotation ---- ...
        here the text representing the tokens 'A, D, I, S, A' will be one predicted annotation (mention).
        Assumes that the 'O' label means outside of mention.

        Requires predicted_label[0].value for each token to be set.
        """
        # TODO
        warnings.warn('annotations actually means entities. This method and related attributes will soon be renamed')

        for part_id, part in self.partids_with_parts():
            for sentence in part.sentences:
                index = 0
                while index < len(sentence):
                    token = sentence[index]
                    confidence_values = []
                    if token.predicted_labels[0].value is not 'O':
                        start = token.start
                        confidence_values.append(token.predicted_labels[0].confidence)
                        while index + 1 < len(sentence) \
                                and sentence[index + 1].predicted_labels[0].value not in ('O', 'B', 'A'):
                            token = sentence[index + 1]
                            confidence_values.append(token.predicted_labels[0].confidence)
                            index += 1
                        end = token.start + len(token.word)
                        confidence = aggregator_function(confidence_values)
                        part.predicted_annotations.append(Entity(class_id, start, part.text[start:end], confidence))
                    index += 1

        return self


    def form_predicted_relations(self):
        """
        Populates part.predicted_relations with a list of Relation objects
        based on the values of the field target for each edge.

        Each Relation object denotes a relationship between two entities
        of (usually) different classes. Each relation is given by a relation type.

        Requires edge.target to be set for each edge.
        """

        for part in self.parts():
            for e in part.edges:

                if e.target == 1:
                    r = Relation(e.relation_type, e.entity1, e.entity2)
                    part.predicted_relations.append(r)

        return self


    def validate_annotation_offsets(self):
        """
        Helper function to validate that the annotation offsets match the annotation text.
        Mostly used as a sanity check and to make sure GNormPlus works as intentded.
        """
        # TODO
        warnings.warn('annotations actually means entities. This method and related attributes will soon be renamed')

        for part in self.parts():
            for ann in part.predicted_annotations:
                if not ann.text == part.text[ann.offset:ann.offset+len(ann.text)]:
                    warnings.warn('the offsets do not match in {}'.format(ann))


    def generate_top_stats_array(self, class_id, top_nr=10, is_alpha_only=False):
        """
        An array for most occuring words.
        :param top_nr: how many top words are shown
        """
        # NOTE ambiguos words?

        raw_dict = {}

        for ann in self.annotations():
            for word in ann.text.split(" "):
                lc_word = word.lower()
                if lc_word.isalpha() and ann.class_id == class_id:
                    if lc_word not in raw_dict:
                        raw_dict[lc_word] = 1
                    else:
                        raw_dict[lc_word] += 1

        # sort by highest number
        sort_dict = OrderedDict(sorted(raw_dict.items(), key=lambda x: x[1], reverse=True))
        print(json.dumps(sort_dict, indent=4))


    def clean_subclasses(self):
        """
        cleans all subclass by setting them to = False
        """
        for ann in self.annotations():
            ann.subclass = False


    def get_size_chars(self):
        """
        :return: total number of chars in this dataset
        """
        return sum(doc.get_size() for doc in self.documents.values())


    def __repr__(self):
        # TODO
        warnings.warn('annotations actually means entities. This method and related attributes will soon be renamed')

        return "Dataset({0} documents and {1} annotations)".format(len(self.documents),
                                                                   sum(1 for _ in self.annotations()))


    def __str__(self):
        second_part = "\n".join(
            ["---DOCUMENT---\nDocument ID: '" + pmid + "'\n" + str(doc) for pmid, doc in self.documents.items()])
        return "----DATASET----\nNr of documents: " + str(len(self.documents)) + ', Nr of chars: ' + str(
            self.get_size_chars()) + '\n' + second_part


    def extend_dataset(self, other):
        """
        Does run on self and returns nothing. Extends the self-dataset with other-dataset.
        Each Document-ID that already exists in self-dataset gets skipped to add.
        :type other: nalaf.structures.data.Dataset
        """
        for key in other.documents:
            if key not in self.documents:
                self.documents[key] = other.documents[key]


    def prune_empty_parts(self):
        """
        deletes all the parts that contain no entities at all
        """
        for doc_id, doc in self.documents.items():
            part_ids_to_del = []
            for part_id, part in doc.parts.items():
                if len(part.annotations) == 0:
                    part_ids_to_del.append(part_id)
            for part_id in part_ids_to_del:
                del doc.parts[part_id]

    def prune_filtered_sentences(self, filterin=(lambda _: False), percent_to_keep=0):
        """
        Depends on labeler
        """
        empty_sentence = lambda s: all(t.original_labels[0].value == 'O' for t in s)

        for part in self.parts():
            tmp = []
            tmp_ = []
            for index, sentence in enumerate(part.sentences):
                do_use = not empty_sentence(sentence) or filterin(part.sentences_[index]) or random.uniform(0, 1) < percent_to_keep
                if do_use:
                    tmp.append(sentence)
                    tmp_.append(part.sentences_[index])
            part.sentences = tmp
            part.sentences_ = tmp_

    def prune_sentences(self, percent_to_keep=0):
        """
        * keep all sentences that contain  at least one mention
        * keep a random selection of the rest of the sentences

        :param percent_to_keep: what percentage of the sentences with no mentions to keep
        :type percent_to_keep: float
        """
        for part in self.parts():
            # find which sentences have at least one mention
            sentences_have_ann = [any(sentence[0].start <= ann.offset < ann.offset + len(ann.text) <= sentence[-1].end
                                      for ann in part.annotations)
                                  for sentence in part.sentences]
            if any(sentences_have_ann):
                # choose a certain percentage of the ones that have no mention
                false_indices = [index for index in range(len(part.sentences)) if not sentences_have_ann[index]]
                chosen = random.sample(false_indices, round(percent_to_keep*len(false_indices)))

                # keep the sentence if it has a mention or it was chosen randomly
                part.sentences = [sentence for index, sentence in enumerate(part.sentences)
                                  if sentences_have_ann[index] or index in chosen]
            else:
                part.sentences = []

    def delete_subclass_annotations(self, subclasses, predicted=True):
        """
        Method does delete all annotations that have subclass.
        Will not delete anything if not specified that particular subclass.
        :param subclasses: one ore more subclasses to delete
        :param predicted: if False it will only consider Part.annotations array and not Part.pred_annotations
        """
        # TODO
        warnings.warn('annotations actually means entities. This method and related attributes will soon be renamed')

        # if it is not a list, create a list of 1 element
        if not hasattr(subclasses, '__iter__'):
            subclasses = [subclasses]

        for part in self.parts():
            part.annotations = [ann for ann in part.annotations if ann.subclass not in subclasses]
            if predicted:
                part.predicted_annotations = [ann for ann in part.predicted_annotations
                                              if ann.subclass not in subclasses]

    def _cv_kfold_split(l, k, fold, validation_set=True):
        total_size = len(l)
        sub_size = round(total_size / k)

        def folds(k, fold):
            return [i % k for i in list(range(fold, fold + k))]

        subsamples = folds(k, fold)
        training = subsamples[0:k-2]
        validation = subsamples[k-2:k-1]
        test = subsamples[k-1:k]

        if validation_set:
            test = []
        else:
            training += validation
            validation = []

        def create_set(subsample_indexes):
            ret = []
            for sub in subsample_indexes:
                start = sub_size * sub
                end = (start + sub_size) if sub != (k-1) else total_size  # k-1 is the last subsample index
                ret += l[start:end]
            return ret

        return (create_set(training), create_set(validation), create_set(test))


    def cv_kfold_split(self, k, fold, validation_set=True):
        keys = list(sorted(self.documents.keys()))
        random.seed(2727)
        random.shuffle(keys)

        def create_dataset(sett):
            ret = Dataset()
            for elem in sett:
                ret.documents[elem] = self.documents[elem]
            return ret

        training, validation, test = Dataset._cv_kfold_split(keys, k, fold, validation_set)

        return (create_dataset(training), create_dataset(validation), create_dataset(test))


    def fold_nr_split(self, n, fold_nr):
        """
        Sugar syntax for next(self.cv_split(n, fold_nr), None)
        """
        return next(self.cv_split(n, fold_nr), None)

    def cv_split(self, n=5, fold_nr=None):
        """
        Returns a generator of N tuples train & test random splits
        according to the standard N-fold cross validation scheme.
        If fold_nr is given, a generator of a single fold is returned (to read as next(g, None))
        Note that fold_nr is 0-indexed. That is, for n=5 the folds 0,1,2,3,4 exist.

        :param n: number of folds
        :type n: int

        :param fold_nr: optional, single fold number to return, 0-indexed.

        :return: a list of N train datasets and N test datasets
        :rtype: (list[nalaf.structures.data.Dataset], list[nalaf.structures.data.Dataset])
        """
        keys = list(sorted(self.documents.keys()))
        random.seed(2727)
        random.shuffle(keys)

        fold_size = int(len(keys) / n)

        def get_fold(fold_nr):
            """
            fold_nr starts in 0
            """
            start = fold_nr * fold_size
            end = start + fold_size
            test_keys = keys[start:end]
            train_keys = [key for key in keys if key not in test_keys]

            test = Dataset()
            for key in test_keys:
                test.documents[key] = self.documents[key]

            train = Dataset()
            for key in train_keys:
                train.documents[key] = self.documents[key]

            return train, test

        if fold_nr:
            assert(0 <= fold_nr < n)
            yield get_fold(fold_nr)
        else:
            for fold_nr in range(n):
                yield get_fold(fold_nr)


    def percentage_split(self, percentage=0.66):
        """
        Splits the dataset randomly into train and test dataset
        given the size of the train dataset in percentage.

        :param percentage: the size of the train dataset between 0.0 and 1.0
        :type percentage: float

        :return train dataset, test dataset
        :rtype: (nalaf.structures.data.Dataset, nalaf.structures.data.Dataset)
        """
        keys = list(sorted(self.documents.keys()))
        # 2727 is an arbitrary number when Alex was drunk one day, and it's just to have reliable order in data folds randomization
        random.seed(2727)
        random.shuffle(keys)

        len_train = int(len(keys) * percentage)
        train_keys = keys[:len_train]
        test_keys = keys[len_train:]

        train = Dataset()
        test = Dataset()

        for key in test_keys:
            test.documents[key] = self.documents[key]
        for key in train_keys:
            train.documents[key] = self.documents[key]

        return train, test

    def stratified_split(self, percentage=0.66):
        """
        Splits the dataset randomly into train and test dataset
        given the size of the train dataset in percentage.

        Additionally it tries to keep the distribution of entities
        in terms of subclass similar in both sets.

        :param percentage: the size of the train dataset between 0.0 and 1.0
        :type percentage: float

        :return train dataset, test dataset
        :rtype: (nalaf.structures.data.Dataset, nalaf.structures.data.Dataset)
        """
        from collections import Counter
        from itertools import groupby
        train = Dataset()
        test = Dataset()

        strat = [(doc_id, Counter(ann.subclass for part in doc for ann in part.annotations))
                 for doc_id, doc in self.documents.items()]
        strat = sorted(strat, key=lambda x: (x[1].get(0, 0), x[1].get(1, 0), x[1].get(2, 0), x[0]))

        switch = 0
        for _, group in groupby(strat, key=lambda x: x[1]):
            group = list(group)
            if len(group) == 1:
                tmp = train if switch else test
                tmp.documents[group[0][0]] = self.documents[group[0][0]]
                switch = 1 - switch
            else:
                len_train = round(len(group) * percentage)
                random.seed(2727)
                random.shuffle(group)

                for key in group[:len_train]:
                    train.documents[key[0]] = self.documents[key[0]]
                for key in group[len_train:]:
                    test.documents[key[0]] = self.documents[key[0]]

        return train, test


class Document:
    """
    Class representing a single document, for example an article from PubMed.

    :type parts: dict
    """

    def __init__(self):
        self.parts = OrderedDict()
        """
        parts the document consists of, encoded as a dictionary
        where the key (string) is the id of the part
        and the value is an instance of Part
        """

    def __eq__(self, other):
        return self.get_size() == other.get_size()

    def __lt__(self, other):
        return self.get_size() - other.get_size() < 0

    def __iter__(self):
        """
        when iterating through the document iterate through each part
        """
        for part_id, part in self.parts.items():
            yield part

    def __repr__(self):
        if self.get_text() == self.get_body():
            return 'Document(Size: {}, Text: "{}", Annotations: "{}")'.format(len(self.parts), self.get_text(),
                                                                                 self.get_unique_mentions())
        else:
            return 'Document(Size: {}, Title: "{}", Text: "{}", Annotations: "{}")'.format(len(self.parts),
                                                                                               self.get_title(),
                                                                       self.get_text(), self.get_unique_mentions())

    def __str__(self):
        partslist = ['--PART--\nPart ID: "' + partid + '"\n' + str(part) + "\n" for partid, part in self.parts.items()]
        second_part = "\n".join(partslist)
        return 'Size: ' + str(self.get_size()) + ", Title: " + self.get_title() + '\n' + second_part

    def key_value_parts(self):
        """yields iterator for partids"""
        for part_id, part in self.parts.items():
            yield part_id, part

    def get_unique_mentions(self):
        """:return: set of all mentions (standard + natural language)"""
        mentions = []
        for part in self:
            for ann in part.annotations:
                mentions.append(ann.text)

        return set(mentions)

    def unique_relations(self, rel_type, predicted=False):
        """
        :param predicted: iterate through predicted relations or true relations
        :type predicted: bool
        :return: set of all relations (ignoring the text offset and
        considering only the relation text)
        """

        relations = []
        for part in self:
            if predicted:
                relation_list = part.predicted_relations
            else:
                relation_list = part.relations

            for rel in relation_list:
                entity1, relation_type, entity2 = rel.get_relation_without_offset()

                if relation_type == rel_type:

                    if entity1 < entity2:
                        relation_string = entity1 + ' ' + relation_type + ' ' + entity2
                    else:
                        relation_string = entity2 + ' ' + relation_type + ' ' + entity1

                    if relation_string not in relations:
                        relations.append(relation_string)

        ret = set(relations)

        return ret


    def relations(self):
        """  helper function for providing an iterator of relations on document level """
        for part in self.parts.values():
            for rel in part.relations:
                yield rel

    def purge_false_relationships(self):
        """
        purging false relationships (that do not return true if validating themselves)
        :return:
        """
        for part in self.parts:
            part.relations[:] = [x for x in part.relations if x.validate_itself(part)]

    def get_size(self):
        """returns nr of chars including spaces between parts"""
        return sum(len(x.text) + 1 for x in self.parts.values()) - 1

    def get_title(self):
        """:returns title of document as str"""
        if len(self.parts.keys()) == 0:
            return ""
        else:
            return list(self.parts.values())[0].text

    def get_text(self):
        """
        Gives the whole text concatenated with spaces in between.
        :return: string
        """
        text = ""

        _length = self.get_size()

        for p in self.parts.values():
            text += "{0} ".format(p.text)
        return text.strip()

    def get_body(self):
        """
        :return: Text without title. No '\n' and spaces between parts.
        """
        text = ""
        size = len(self.parts)
        for i, (_, part) in enumerate(self.parts.items()):
            if i > 0:
                if i < size - 1:
                    text += part.text.strip() + " "
                else:
                    text += part.text.strip()
        return text

    def overlaps_with_mention2(self, start, end):
        """
        Checks for overlap with given 2 nrs that represent start and end position of any corresponding string.
        :param start: index of first char (offset of first char in whole document)
        :param end: index of last char (offset of last char in whole document)
        """
        print_verbose('Searching for overlap with a mention.')
        Entity.equality_operator = 'exact_or_overlapping'
        query_ann = Entity(class_id='', offset=start, text=(end - start + 1) * 'X')
        print_debug(query_ann)
        offset = 0
        for part in self.parts.values():
            print_debug('Query: Offset =', offset, 'start char =', query_ann.offset, 'start char + len(ann.text) =',
                        query_ann.offset + len(query_ann.text), 'params(start, end) =',
                        "({0}, {1})".format(start, end))
            for ann in part.annotations:
                offset_corrected_ann = Entity(class_id='', offset=ann.offset + offset, text=ann.text)
                if offset_corrected_ann == query_ann:
                    print_verbose('Found annotation:', ann)
                    return True
                else:
                    print_debug(
                        "Current(offset: {0}, offset+len(text): {1}, text: {2})".format(offset_corrected_ann.offset,
                                                                                        offset_corrected_ann.offset + len(
                                                                                            offset_corrected_ann.text),
                                                                                        offset_corrected_ann.text))
            offset += len(part.text)
        return False

    def overlaps_with_mention(self, *span, annotated=True):
        """
        Checks for overlap at position charpos with another mention.
        """
        offset = 0

        if len(span) == 2:
            start, end = span
        else:
            start, end = span[0]
        # todo check again with *span and unpacking

        print_debug("===TEXT===\n{0}\n".format(self.get_text()))

        for pid, part in self.parts.items():
            print_debug("Part {0}: {1}".format(pid, part))
            if annotated:
                annotations = part.annotations
            else:
                annotations = part.predicted_annotations
            for ann in annotations:
                print_debug(ann)
                print_debug("TEXT:".ljust(10) + part.text)
                print_debug("QUERY:".ljust(10) + "o" * (start - offset) + "X" * (end - start + 1) + "o" * (
                    len(part.text) - end + offset - 1))
                print_debug("CURRENT:".ljust(10) + ann.text.rjust(ann.offset + len(ann.text), 'o') + 'o' * (
                        len(part.text) - ann.offset + len(ann.text) - 2))
                if start < ann.offset + offset + len(ann.text) and ann.offset + offset <= end:
                    print_verbose('=====\nFOUND\n=====')
                    print_verbose("TEXT:".ljust(10) + part.text)
                    print_verbose("QUERY:".ljust(10) + "o" * (start - offset) + "X" * (end - start + 1) + "o" * (
                        len(part.text) - end + offset - 1))
                    print_verbose("FOUND:".ljust(10) + ann.text.rjust(ann.offset + len(ann.text), 'o') + 'o' * (
                        ann.offset + len(ann.text) - 1))
                    return ann
            offset += len(part.text) + 1
        print_verbose('=========\nNOT FOUND\n=========')
        print_verbose(
            "QUERY:".ljust(10) + "o" * start + "X" * (end - start + 1) + "o" * (offset - end - 2))
        print_verbose("TEXT:".ljust(10) + self.get_text())
        print_debug()
        return False


class Part:
    """
    Represent chunks of text grouped in the document that for some reason belong together.
    Each part hold a reference to the annotations for that chunk of text.

    :type text: str
    :type sentences_: list[str]
    :type sentences: list[list[Token]]
    :type annotations: list[Entity]
    :type predicted_annotations: list[Entity]
    :type is_abstract: bool
    """

    def __init__(self, text, is_abstract=True):
        self.text = text
        """the original raw text that the part is consisted of"""

        self.sentences_ = []
        """the text sentences previous tokenization"""

        self.sentences = [[]]
        """
        a list sentences where each sentence is a list of tokens
        derived from text by calling Splitter and Tokenizer
        """

        self.annotations = []
        """the entity of the chunk of text as populated by a call to Annotator"""
        # TODO
        warnings.warn('annotations actually means entities. This method and related attributes will soon be renamed')

        self.predicted_annotations = []
        """
        a list of predicted entities as populated by a call to form_predicted_annotations()
        this represent the prediction on a mention label rather then on a token level
        """
        # TODO
        warnings.warn('annotations actually means entities. This method and related attributes will soon be renamed')

        self.relations = []
        """
        a list of relations that represent a connection between 2 annotations e.g. mutation mention and protein,
        where the mutation occurs inside
        """

        self.predicted_relations = []
        """a list of predicted relations as populated by a call to form_predicted_relations()"""

        self.edges = []
        """a list of possible relations between any two entities in the part"""

        self.is_abstract = is_abstract
        """whether the part is the abstract of the paper"""

        self.sentence_parse_trees = []
        """the parse trees for each sentence stored as a string. TODO this may be too relna-specific"""

        self.tokens = []
        # TODO what's this?


    def get_sentence_string_array(self):
        """ :returns an array of string in which each index contains one sentence in type string with spaces between tokens """

        if len(self.sentences) == 0 or len(self.sentences[0]) == 0:
            return self.sentences_
        else:
            return_array = []
            for sentence_array in self.sentences:
                new_sentence = ""
                for token in sentence_array:
                    new_sentence += token.word + " "

                return_array.append(new_sentence.rstrip())  # to delete last space
            return return_array


    def get_sentence_index_for_annotation(self, annotation):

        for sentence_index, sentence in enumerate(self.sentences):
            assert sentence != [[]] and sentence != [], "The sentences have not been splitted/defined yet"

            sentence_start = sentence[0].start
            sentence_end = sentence[-1].end

            if sentence_start <= annotation.offset < sentence_end:
                return sentence_index

        assert False, ("The annotation did not (and should) have an associated sentence. Ann: " + str(annotation))


    def get_entity(self, start_offset, raise_exception_on_incosistencies=True):
        """
        Retrieves entity object from a list of annotations based on start_offset value.
        """
        found_list = list(filter(lambda ann: ann.offset == start_offset, self.annotations))
        length = len(found_list)

        if length == 0:
            if (raise_exception_on_incosistencies):
                raise Exception("Entity with offset {} was expected and yet was not found".format(str(start_offset)))
            else:
                return None

        elif length == 1:
            return found_list[0]

        else:
            if (raise_exception_on_incosistencies):
                raise Exception("As of now, Part's should not have multiple entities with _same_ start_offset: {} -- found: {}, list: \n\t{}".format(str(start_offset), str(length), ("\n\t".join((str(e) for e in found_list)))))
            else:
                return found_list[0]


    def get_entities_in_sentence(self, sentence_id, entity_classId):
        """
        get entities of a particular type in a particular sentence

        :param sentence_id: sentence number in the part
        :type sentence_id: int
        :param entity_classId: the classId of the entity
        :type entity_classId: str
        """
        sentence = self.sentences[sentence_id]
        start = sentence[0].start
        end = sentence[-1].end
        entities = []
        for annotation in self.annotations:
            if start <= annotation.offset < end and annotation.class_id == entity_classId:
                entities.append(annotation)
        return entities


    def percolate_tokens_to_entities(self, annotated=True):
        """
        if entity start and token start, and entity end and token end match,
        store tokens directly.
        if entity start and token start or entity end and token end don't match
        store the nearest entity having index just before for the start of the
        entity and just after for the end of the entity
        """
        for entity in chain(self.annotations, self.predicted_annotations):
            entity.tokens = []
            entity_end = entity.offset + len(entity.text)
            for sentence in self.sentences:
                for token in sentence:
                    if entity.offset <= token.start < entity_end or \
                        token.start <= entity.offset < token.end:
                        entity.tokens.append(token)


    # TODO move to edge features
    def calculate_token_scores(self):
        """
        calculate score for each entity based on a simple heuristic of which
        token is closest to the root based on the dependency tree.
        """
        not_tokens = []
        important_dependencies = ['det', 'amod', 'appos', 'npadvmod', 'compound',
                'dep', 'with', 'nsubjpass', 'nsubj', 'neg', 'prep', 'num', 'punct']
        for sentence in self.sentences:
            for token in sentence:
                if token.word not in not_tokens:
                    token.features['score'] = 1
                if token.features['dependency_from'][0].word not in not_tokens:
                    token.features['dependency_from'][0].features['score'] = 1

            done = False
            counter = 0

            while(not done):
                done = True
                for token in sentence:
                    dep_from = token.features['dependency_from'][0]
                    dep_to = token
                    dep_type = token.features['dependency_from'][1]

                    if dep_type in important_dependencies:
                        if dep_from.features['score'] <= dep_to.features['score']:
                            dep_from.features['score'] = dep_to.features['score'] + 1
                            done = True
                counter += 1
                if counter > 20:
                    break


    def set_head_tokens(self):
        """
        set head token for each entity based on the scores for each token
        """
        for token in self.tokens:
            if token.features['score'] is None:
                token.features['score'] = 1

        for entity in chain(self.annotations, self.predicted_annotations):
            if len(entity.tokens) == 1:
                entity.head_token = entity.tokens[0]
            else:
                entity.head_token = max(entity.tokens, key=lambda token: token.features['score'])

    def __iter__(self):
        """
        when iterating through the part iterate through each sentence
        """
        return iter(self.sentences)

    def __repr__(self):
        return "Part(is abstract = {abs}, len(sentences) = {sl}, ' \
        'len(anns) = {al}, len(pred anns) = {pl}, ' \
        'len(rels) = {rl}, len(pred rels) = {prl}, ' \
        'text = \"{self.text}\")".format(
            self=self, sl=len(self.sentences),
            al=len(self.annotations), pl=len(self.predicted_annotations),
            rl=len(self.relations), prl=len(self.predicted_relations),
            abs=self.is_abstract)

    def __str__(self):
        entities_string = "\n".join([str(x) for x in self.annotations])
        pred_entities_string = "\n".join([str(x) for x in self.predicted_annotations])
        relations_string = "\n".join([str(x) for x in self.relations])
        pred_relations_string = "\n".join([str(x) for x in self.predicted_relations])
        if not entities_string:
            entities_string = "[]"
        if not pred_entities_string:
            pred_entities_string = "[]"
        if not relations_string:
            relations_string = "[]"
        if not pred_relations_string:
            pred_relations_string = "[]"
        return 'Is Abstract: {abstract}\n-Text-\n"{text}"\n-Entities-\n{annotations}\n' \
               '-Predicted entities-\n{pred_annotations}\n' \
               '-Relations-\n{relations}\n' \
               '-Predicted relations-{pred_relations}'.format(
                        text=self.text, annotations=entities_string,
                        pred_annotations=pred_entities_string, relations=relations_string,
                        pred_relations=pred_relations_string, abstract=self.is_abstract)

    def get_size(self):
        """ just returns number of chars that this part contains """
        # OPTIONAL might be updated in order to represent entities and such as well
        return len(self.text)


class Edge:
    """
    Represent an edge - a possible relation between two named entities.

    Note:
        The same_ (part or sentence_id) are helper / sugar field for convenience
        Their use is discourage and ideally the library would throw a warning

        The best best solution would be to be able to retrieve the part and sentence_id from the entities directly
    """

    def __init__(self, relation_type, entity1, entity2, e1_part, e2_part, e1_sentence_id, e2_sentence_id):
        self.relation_type = relation_type
        self.entity1 = entity1
        self.entity2 = entity2

        self.e1_part = e1_part
        """The part in which entity1 is contained"""

        self.e2_part = e2_part
        """The part in which entity2 is contained"""

        assert self.e1_part == self.e2_part, "As of now, only relationships within a _same_ part are allowed"

        self.same_part = self.e1_part

        self.e1_sentence_id = e1_sentence_id
        """The index of the sentence mentioning entity1 (contain in its corresponding part)"""

        self.e2_sentence_id = e2_sentence_id
        """The index of the sentence mentioning entity2 (contain in its corresponding part)"""

        self.same_sentence_id = AssertionError("The assummed _same_ sentences, are actually different: {} vs {}".format(self.e1_sentence_id, self.e2_sentence_id))

        if (self.e1_sentence_id == self.e2_sentence_id):
            self.same_sentence_id = self.e1_sentence_id

        self.features = {}
        """
        a dictionary of features for the edge
        each feature is represented as a key value pair:
        """

        self.target = None
        """class of the edge -- ASSUMED to be in [-1, +1] or None when not defined"""
        # TODO we should much more carefully take care of its type, and whether it could even contain other values
        # As of now, it seems to devependant on `svmlight.py`


    def is_relation(self):
        """
        check if the edge is present in part.relations.
        :rtype: bool
        """
        # TODO change the equals method in Relation appropriately not to do thi bullshit

        relation_1 = Relation(self.relation_type, self.entity1, self.entity2)
        relation_2 = Relation(self.relation_type, self.entity2, self.entity1)

        # TODO, yes, we are aware that we also have self.same_part. However, ideally here we do not use that variable
        assert(self.e1_part == self.e2_part)
        self_part = self.e1_part

        for relation in self_part.relations:
            if relation_1 == relation:
                return True
            if relation_2 == relation:
                return True

        return False


    def __repr__(self):
        return 'Edge between "{0}" and "{1}" of the type "{2}".'.format(self.entity1.text, self.entity2.text, self.relation_type)


class Token:
    """
    Represent a token - the smallest unit on which we perform operations.
    Usually one token represent one word from the document.

    :type word: str
    :type original_labels: list[Label]
    :type predicted_labels: list[Label]
    :type features: FeatureDictionary
    """

    def __init__(self, word, start):
        self.word = word
        """string value of the token, usually a single word"""
        self.start = start
        """start offset of the token in the original text"""
        self.end = self.start + len(self.word)
        """end offset of the token in the original text"""
        self.original_labels = None
        """the original labels for the token as assigned by some implementation of Labeler"""
        self.predicted_labels = None
        """the predicted labels for the token as assigned by some learning algorightm"""
        self.features = FeatureDictionary()
        """
        a dictionary of features for the token
        each feature is represented as a key value pair:
        * [string], [string] pair denotes the feature "[string]=[string]"
        * [string], [float] pair denotes the feature "[string]:[float] where the [float] is a weight"
        """

    def is_entity_part(self, part):
        """
        check if the token is part of an entity
        :return bool:
        """
        for entity in part.annotations:
            if self.start <= entity.offset < self.end:
                return True
        return False

    def get_entity(self, part):
        """
        if the token is part of an entity, return the entity else return None
        :param part: an object of type Part in which to search for the entity.
        :type part: nalaf.structures.data.Part
        :return nalaf.structures.data.Entity or None
        """
        for entity in part.annotations:
            if self.start <= entity.offset < self.end:
                # entity.offset <= self.start < entity.offset + len(entity.text):
                return entity
        return None

    # TODO review this method. This was added for relna. It is now also used in LocText
    def masked_text(self, part):
        """
        if token is part of an entity, return the entity class id, otherwise
        return the token word itself.
        :param part: an object of type Part in which to search for the entity.
        :type part: nalaf.structures.data.Part
        :return str
        """
        for entity in part.annotations:
            if self.start <= entity.offset < self.end:  # or \
                # entity.offset <= self.start < entity.offset + len(entity.text):
                return entity.class_id
        return self.word

    def __repr__(self):
        """
        print calls to the class Token will print out the string contents of the word
        """
        return self.word

    def __eq__(self, other):
        """
        consider two tokens equal if and only if their token words and start
        offsets coincide.
        :type other: nalaf.structures.data.Token
        :return bool:
        """
        if hasattr(other, 'word') and hasattr(other, 'start'):
            if self.word == other.word and self.start == other.start:
                return True
            else:
                return False
        else:
            return False

    def __ne__(self, other):
        """
        :type other: nalaf.structures.data.Token
        :return bool:
        """
        return not self.__eq__(other)


class FeatureDictionary(dict):
    """
    Extension of the built in dictionary with the added constraint that
    keys (feature names) cannot be updated.

    If the key (feature name) doesn't end with "[number]" appends "[0]" to it.
    This is used to identify the position in the window for the feature.

    Raises an exception when we try to add a key that exists already.
    """

    def __setitem__(self, key, value):
        if key in self:
            raise KeyError('feature name "{}" already exists'.format(key))
        else:
            if not re.search('\[-?[0-9]+\]$', key):
                key += '[0]'
            dict.__setitem__(self, key, value)


class Entity:
    """
    Represent a single annotation, that is denotes a span of text which represents some entity.

    :type class_id: str
    :type offset: int
    :type text: str
    :type subclass: int
    :type confidence: float
    :type normalisation_dict: dict
    :type normalized_text: str
    :type tokens: list[nalaf.structures.data.Token]
    :type head_token: nalaf.structures.data.Token
    """
    def __init__(self, class_id, offset, text, confidence=1):
        self.class_id = class_id
        """the id of the class or entity that is annotated"""
        self.offset = offset
        """the offset marking the beginning of the annotation in regards to the Part this annotation is attached to."""
        self.text = text
        """the text span of the annotation"""
        self.subclass = False
        """
        int flag used to further subdivide classes based on some criteria
        for example for mutations (MUT_CLASS_ID): 0=standard, 1=natural language, 2=semi standard
        """
        self.confidence = confidence
        """aggregated mention level confidence from the confidence of the tokens based on some aggregation function"""
        self.normalisation_dict = {}
        """ID in some normalization database of the normalized text for the annotation if normalization was performed"""
        self.normalized_text = ''
        """the normalized text for the annotation if normalization was performed"""
        self.tokens = []
        """
        the tokens in each entity
        TODO Note that tokens are already within sentences. You should use those by default.
        This list of tokens may be deleted. See: https://github.com/Rostlab/nalaf/issues/167
        """
        self.head_token = None
        """the head token for the entity. Note: this is not necessarily the first token, just the head of the entity as declared by parsing (see relna)"""


    equality_operator = 'exact'
    """
    determines when we consider two entities to be equal
    can be "exact" or "overlapping" or "exact_or_overlapping"
    """


    def __repr__(self):
        norm_string = ''

        if self.normalisation_dict:
            norm_string = ', Normalisation Dict: {0}, Normalised text: "{1}"'.format(self.normalisation_dict, self.normalized_text)

        return 'Entity(ClassID: "{self.class_id}", Offset: {self.offset}, ' \
               'Text: "{self.text}", SubClass: {self.subclass}, ' \
               'Confidence: {self.confidence}{norm})'.format(self=self, norm=norm_string)


    def __eq__(self, other):
        # consider them a match only if class_id matches
        # TODO implement test case for edge cases in overlap and exact
        if self.class_id == other.class_id:
            exact = self.offset == other.offset and self.text == other.text
            overlap = self.offset < (other.offset + len(other.text)) and (self.offset + len(self.text)) > other.offset

            if self.equality_operator == 'exact':
                return exact
            elif self.equality_operator == 'overlapping':
                # overlapping means only the case where we have an actual overlap and not exact match
                return not exact and overlap
            elif self.equality_operator == 'exact_or_overlapping':
                # overlap includes the exact case so just return that
                return overlap
            else:
                raise ValueError('other must be "exact" or "overlapping" or "exact_or_overlapping"')
        else:
            return False


class Label:
    """
    Represents the label associated with each Token.

    :type value: str
    :type confidence: float
    """

    def __init__(self, value, confidence=None):
        self.value = value
        """string value of the label"""
        self.confidence = confidence
        """probability of being correct if the label is predicted"""

    def __repr__(self):
        return self.value


class Relation:
    """
    Represents a relationship between 2 entities.
    """

    def __init__(self, relation_type, entity1, entity2):
        self.class_id = relation_type
        assert entity1 is not None and entity2 is not None, "Some of the entities are None"

        self.entity1 = entity1
        self.entity2 = entity2


    def __repr__(self):
        return 'Relation(Class ID:"{self.class_id}", entity1:"{str(self.entity1)}", entity2:"{str(self.entity2)}")'.format(self=self)


    def get_relation_without_offset(self):
        """:return string with entity1 and entity2 separated by relation type"""
        return (self.entity1.text, self.class_id, self.entity2.text)


    def validate_itself(self, part):
        """
        validation of itself with entities and the text
        :param part: the part where this relation is saved inside
        :type part: nalaf.structures.data.Part
        :return: bool
        """
        first = False
        second = False

        for ann in chain(part.annotations, part.predicted_annotations):

            if ann.offset == self.entity1.offset and ann.text == self.entity1.text:
                first = True
            if ann.offset == self.entity2.øffset and ann.text == self.entity2.text:
                second = True
            if first and second:
                return True

        return False


    def __eq__(self, other):
        """
        consider two relations equal if and only if all their parameters match
        :type other: nalaf.structures.data.Relation
        :return bool:
        """

        if other is not None:
            # TODO CAUTION (https://github.com/juanmirocks/LocText/issues/6) this may have terrible consequences
            return self.__dict__ == other.__dict__

        else:
            return False


    def __ne__(self, other):
        """
        :type other: nalaf.structures.data.Relation
        :return bool:
        """
        if other is not None:
            return not self.__dict__ == other.__dict__
        else:
            return False
