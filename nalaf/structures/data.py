from collections import OrderedDict
import json
import random
import re
from nalaf.utils.qmath import arithmetic_mean
from nalaf import print_debug, print_verbose
import warnings
from itertools import chain
from collections import Counter


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


    def entities(self):
        """
        Yield all entities of the dataset.

        :rtype: collections.Iterable[Entity]
        """

        for part in self.parts():
            for annotation in part.annotations:
                yield annotation


    def annotations(self):
        warnings.warn('Use `self.entities` instead', DeprecationWarning)
        return self.entities()


    def predicted_entities(self):
        """
        Yield all predicted entities of the dataset.

        :rtype: collections.Iterable[Entity]
        """
        for part in self.parts():
            for annotation in part.predicted_annotations:
                yield annotation


    def predicted_annotations(self):
        warnings.warn('Use `self.predicted_entities` instead', DeprecationWarning)
        return self.predicted_entities()


    def relations(self):
        """
        Yield all relations of the Dataset.

        :rtype: collections.Iterable[Relation]
        """
        for part in self.parts():
            for rel in part.relations:
                yield rel


    def predicted_relations(self):
        """
        Yield all predicted relations of the dataset.

        :rtype: collections.Iterable[Relation]
        """
        for part in self.parts():
            for relation in part.predicted_relations:
                yield relation


    def plausible_relations_from_generated_edges(self):
        """
        Yield only the real relations that are obtainable from the corpus-generated edges.
        """
        for edge in self.edges():
            relation = edge.get_relation_if_is_real()
            if relation is not None:
                yield relation


    def compute_stats_relations_distances(self, relation_type, entity_map_fun=None, relation_accept_fun=None):
        """
        Returns a counter of the relationships distances.

        The relationships are mapped to unique strings as determined by entity_map_fun (see `map_relations`).

        The minimal distance of the mapped relations with same map key is used.
        """

        if next(self.predicted_relations(), None) is None:
            raise AssertionError("Relations for the corpus must have been predicted to fully exhaust the search")

        if entity_map_fun is None:
            entity_map_fun = Entity.__repr__

        counter_nums = Counter()

        for doc in self:
            # Group relationships at the document level (not part level, not corpus level)
            doc_relations = {}

            doc_relations = doc.map_relations(use_predicted=False, relation_type=relation_type, entity_map_fun=entity_map_fun)
            pred_doc_relations = doc.map_relations(use_predicted=True, relation_type=relation_type, entity_map_fun=entity_map_fun)

            for pred_key, pred_rels_with_distances in pred_doc_relations.items():

                if pred_key in doc_relations:
                    doc_relations[pred_key] += pred_rels_with_distances

                if relation_accept_fun is not None:
                    for real_key in doc_relations:
                        if relation_accept_fun(real_key, pred_key):
                            doc_relations[real_key] += pred_rels_with_distances

            for rel_key, rels_with_distances in doc_relations.items():
                rel, min_distance_for_unique_key = min(rels_with_distances, key=lambda reldist_tuple: reldist_tuple[1])
                counter_nums.update(["D" + str(min_distance_for_unique_key)])

        total = sum(counter_nums.values())

        counter_percts = Counter({key: (count / total) for (key, count) in counter_nums.items()})

        return (counter_nums, counter_percts)


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
        Yield all this Corpus's edges.

        :rtype: collections.Iterable[Edge]
        """
        for part in self.parts():
            for edge in part.edges:
                yield edge


    def label_edges(self):
        """
        label each edge with its REAL target (no prediction) - whether it is indeed a relation or not
        """
        for edge in self.edges():
            if edge.is_relation():
                edge.real_target = +1
            else:
                edge.real_target = -1


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
                    if token.predicted_labels[0].value != 'O':
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

        Requires edge.pred_target to be set for each edge.
        """

        for part in self.parts():
            for e in part.edges:

                if e.pred_target == +1:
                    r = e.get_potential_relation()
                    part.predicted_relations.append(r)

        return self


    def validate_entity_offsets(self):
        """
        Helper function to validate that the entities offsets match the entity text.
        Use it as a sanity check when writing or reading annotations external entities.
        """

        for docid, doc in self.documents.items():
            for partid, part in doc.parts.items():
                for e in chain(part.annotations, part.predicted_annotations):
                    readable_text = part.text[e.offset:e.offset + len(e.text)]

                    if not e.text == readable_text:
                        warnings.warn('the offsets ({} != {}) do not match in: {}/{}/{}'.format(e.text, part.text[e.offset:e.offset + len(e.text)], docid, partid, e))


    def generate_top_stats_array(self, class_id, top_nr=10, is_alpha_only=False):
        """
        An array for most occuring words.
        :param top_nr: how many top words are shown
        """
        # NOTE ambiguos words?

        raw_dict = {}

        for ann in self.entities():
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
        for ann in self.entities():
            ann.subclass = False


    def get_size_chars(self):
        """
        :return: total number of chars in this dataset
        """
        return sum(doc.get_size() for doc in self.documents.values())


    def __repr__(self):
        def class_repr(class_id):
            return class_id + ": " + str(Counter(e.subclass for e in self.entities() if e.class_id == class_id))

        classes_repr = [class_repr(class_id) for class_id in {e.class_id for e in self.entities()}]

        return "Dataset({} documents and {} entities ({}))".format(len(self.documents), sum(1 for _ in self.entities()), classes_repr)


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
                if len(part.annotations) == 0:  # or part.predict_entities ?
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
        * keep all sentences that contain at least one mention
        * keep a random selection of the rest of the sentences

        :param percent_to_keep: what percentage of the sentences with no mentions to keep
        :type percent_to_keep: float
        """
        for part in self.parts():
            # find which sentences have at least one mention
            sentences_have_ann = [any(sentence[0].start <= ann.offset < ann.offset + len(ann.text) <= sentence[-1].end
                                      for ann in part.annotations)  # pred_annotations ?
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


    @staticmethod
    def _cv_kfold_split(keys, k, fold, validation_set=True):
        """
        Split keys (e.g. doc ids or indexes) into two groups: (training, evaluation)

        if validation_set is True,
            training = strictly training set
            evaluation = validation set
        otherwise
            training = training set + validation set
            evaluation = test set
        """
        total_size = len(keys)
        sub_size = round(total_size / k)

        def folds(k, fold):
            return [i % k for i in list(range(fold, fold + k))]

        subsamples = folds(k, fold)
        training = subsamples[0:k-2]
        validation = subsamples[k-2:k-1]
        test = subsamples[k-1:k]

        if validation_set:
            training = training
            evaluation = validation
        else:
            training = training + validation
            evaluation = test

        def create_keys_set(subsample_indexes):
            ret = []
            for sub in subsample_indexes:
                start = sub_size * sub
                end = (start + sub_size) if sub != (k-1) else total_size  # k-1 is the last subsample index
                ret += keys[start:end]
            return ret

        return (create_keys_set(training), create_keys_set(evaluation))


    @staticmethod
    def _cv_kfold_splits_randomize_keys(keys):
        random.seed(2727)
        random.shuffle(keys)
        return keys


    @staticmethod
    def _cv_kfold_splits_doc_keys_sets(doc_keys, k, validation_set):
        doc_keys = list(sorted(doc_keys))
        doc_keys = __class__._cv_kfold_splits_randomize_keys(doc_keys)

        for fold in range(k):
            training, evaluation = __class__._cv_kfold_split(doc_keys, k, fold, validation_set)
            yield training, evaluation


    def cv_kfold_splits(self, k, validation_set=True):

        def create_dataset(keys):
            ret = Dataset()
            for elem in keys:
                ret.documents[elem] = self.documents[elem]
            return ret

        for training, evaluation in __class__._cv_kfold_splits_doc_keys_sets(self.documents.keys(), k, validation_set):
            yield (create_dataset(training), create_dataset(evaluation))


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


    def entities(self):
        """
        Yield all entities of the Document.

        :rtype: collections.Iterable[Entity]
        """
        for part in self.parts.values():
            for e in part.annotations:
                yield e


    def predicted_entities(self):
        """
        Yield all predicted entities of the Document.

        :rtype: collections.Iterable[Entity]
        """
        for part in self.parts.values():
            for e in part.predicted_annotations:
                yield e


    def relations(self):
        """
        Yield all relations of the Document.

        :rtype: collections.Iterable[Relation]
        """
        for part in self.parts.values():
            for rel in part.relations:
                yield rel


    def predicted_relations(self):
        """
        Yield all predicted relations of the Document.

        :rtype: collections.Iterable[Relation]
        """
        for part in self.parts.values():
            for rel in part.predicted_relations:
                yield rel


    def edges(self):
        """
        Yield all this Document's edges.

        :rtype: collections.Iterable[Edge]
        """
        for part in self:
            for edge in part.edges:
                yield edge


    def get_unique_mentions(self):
        """:return: set of all mentions (standard + natural language)"""
        mentions = []
        for part in self:
            for ann in part.annotations:
                mentions.append(ann.text)

        return set(mentions)


    def map_relations(self, use_predicted, relation_type, entity_map_fun, relations_search_space=None, doc_mapped_relations=None):
        """
        Map all Documents's relations to dictionary of:
        {unique mapped strings --> (list of tuples: (relation with same map key, sentence distance between the related entities))}

        Create a set of the document's relations based on the map function of the relation themselves and the given map
        function for their entities. Relations end up being represented as strings in the set.

        Return: set of strings that represent unique relationsihps
        """

        if doc_mapped_relations is None:
            doc_mapped_relations = {}

        for part in self:
            doc_mapped_relations = part.map_relations(use_predicted, relation_type, entity_map_fun, relations_search_space, doc_mapped_relations)

        return doc_mapped_relations


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


    def get_text(self, separation=" "):
        """
        Gives the whole text concatenated with `separation` parameter string in between.
        :return: string
        """
        return separation.join((p.text for p in self.parts.values()))


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

        self.predicted_annotations = []
        """
        a list of predicted entities as populated by a call to form_predicted_annotations()
        this represent the prediction on a mention label rather then on a token level
        """

        # TODO
        warnings.warn('"annotations" (and "predicted_annotations") are meant to be "entities". This and related attributes will soon be renamed')

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

        # TODO this may be too relna-specific
        self.sentence_parse_trees = []
        """the parse trees for each sentence stored as a string."""


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


    def get_sentence_index_for_annotation(self, entity):

        for sentence_index, sentence in enumerate(self.sentences):
            assert sentence != [[]] and sentence != [], "The sentences have not been splitted/defined yet"

            sentence_start = sentence[0].start
            sentence_end = sentence[-1].end

            if sentence_start <= entity.offset < sentence_end:
                return sentence_index

        assert False, ("The entity did not (and should) have an associated sentence. Ann: " + str(entity))


    def get_entity(self, start_offset, use_pred, raise_exception_on_incosistencies=True):
        """
        Retrieves entity object from a list of annotations based on start_offset value.
        """
        entities = self.annotations if not use_pred else self.predicted_annotations
        found_list = list(filter(lambda ann: ann.offset == start_offset, entities))
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


    def get_any_entities_in_sentence(self, sentence_id, predicted):
        sentence = self.sentences[sentence_id]
        start = sentence[0].start
        end = sentence[-1].end
        entities = self.predicted_annotations if predicted else self.annotations

        ret = {}
        for entity in entities:
            if start <= entity.offset < end:
                updated = ret.get(entity.class_id, [])
                updated.append(entity)
                ret[entity.class_id] = updated
        return ret


    def get_entities_in_sentence(self, sentence_id, entity_classId):
        """
        get entities of a particular type in a particular sentence

        :param sentence_id: sentence number in the part
        :type sentence_id: int
        :param entity_classId: the classId of the entity
        :type entity_classId: str
        """
        import warnings
        warnings.warn('Use rather the method: get_any_entities_in_sentence', DeprecationWarning)

        return self.get_any_entities_in_sentence(sentence_id, predicted=False)[entity_classId]


    def percolate_tokens_to_entities(self, annotated=True):
        """
        if entity start and token start, and entity end and token end match,
        store tokens directly.
        if entity start and token start or entity end and token end don't match
        store the nearest entity having index just before for the start of the
        entity and just after for the end of the entity
        """

        sentences = self.sentences

        for entity in chain(self.annotations, self.predicted_annotations):
            entity.tokens = []
            entity_end = entity.offset + len(entity.text)
            sentence_index = None

            for index, sentence in enumerate(sentences):
                sentence_adjuted = False

                for token in sentence:
                    if entity.offset <= token.start < entity_end or \
                        token.start <= entity.offset < token.end:

                        if sentence_index is not None and sentence_index != index:
                            # entity spanning multiple sentences, --> adjust sentences as it's likely a sentence splitting error
                            # Should happen very seldom
                            # Example: In these cells, Kv8.1 expressed alone remains in intracellular compartments, but it can reach the plasma membrane when it associates with Kv2.2, and it then also forms new types of Kv8.1/Kv2. 2 channels
                            if not sentence_adjuted:
                                print()
                                print("WARNING ADJUST", sentence)
                                print("WARNING ADJUST", sentence_index, index, sentence_adjuted)
                                print("WARNING ADJUST", entity.text, "---", entity)
                                print("WARNING ADJUST", sentences)
                                print("WARNING ADJUST", entity.sentence)
                                sentences[index-1] += sentence
                                del sentences[index]
                                print("WARNING ADJUST", entity.sentence)
                                print("WARNING ADJUST", sentences)
                                print()

                                self.sentences = sentences
                                sentence_adjuted = True

                        entity.tokens.append(token)

                        if sentence_index is None:
                            sentence_index = index
                            entity.sentence = sentence
                            entity.part = self


    @staticmethod
    def get_sentence_roots(sentence, feature_key='is_root'):
        """
        **Depends on** parsers.py :: SpacyParser (dependency parser).

        Gets the roots of a sentence list of tokens.

        Note that the spaCy parser allows multiple roots.
        In this view, a root is a token that does not have any incoming dependency,
        that is, the dependency graph is a tree with multiple roots (more generally, a graph).

        Note that many other parsers enforce to have a sole root by creating a dummy node
        than then connects to the real root nodes (those without real incoming dependencies).
        """
        roots = [token for token in sentence if token.features[feature_key] is True]
        assert len(roots) >= 1, "The sentence contains {} roots (?). Expected: >= 1 -- Sentence: {}".format(len(roots), ' '.join((t.word for t in sentence)))

        return roots


    @staticmethod
    def get_main_verbs(sentence, include_linked_verbs=True, token_map=lambda t: t):

        def search_first_verbs(tokens):
            if len(tokens) == 0:
                return []
            else:
                verbs = [t for t in tokens if t.is_POS_Verb()]

                if len(verbs) > 0:
                    return verbs
                else:
                    return search_first_verbs([dep_to for t in tokens for dep_to, _ in t.features["dependency_to"]])

        roots = Part.get_sentence_roots(sentence)

        verbs = search_first_verbs(roots)

        if include_linked_verbs:
            verbs += [dep_to for t in verbs for dep_to, _ in t.features["dependency_to"] if dep_to.is_POS_Verb()]

        return [token_map(t) for t in verbs]

    @staticmethod
    def is_negated(tokens_path):
        """
        Simple heuristic to derive if a sentence or more generally a path of tokens (e.g. parsing dependency)
        is written affirmatively or negated, as in "Juanmi is awesome" vs. "Juanmi does not give up".

        A path of tokens is negated if it contains an odd number of "neg" (negation) parsed dependencies.
        """
        return (sum(t.features["dep"] == "neg" for t in tokens_path) % 2) != 0


    _FEAT_DEPTH_KEY = 'depth'


    def compute_tokens_depth(self):
        """
        **Depends on** parsers.py :: SpacyParser (dependency parser).


        Computes the depth for every token of every sentence defined as:

        * root tokens have depth == 0
        * other tokens have depth == depth(parent_token) + 1

        The depth of each token is finally saved in the tokens' features with key 'depth'
        """

        for sentence in self.sentences:
            roots = Part.get_sentence_roots(sentence)

            Part._recursive_compute_tokens_depth(current_depth=0, tokens=roots)


    @staticmethod
    def _recursive_compute_tokens_depth(current_depth, tokens):

        if len(tokens) == 0:
            return
        else:
            next_depth_tokens = []
            for token in tokens:

                if Part._FEAT_DEPTH_KEY in token.features:
                    pass  # depth already defined by another and _shorter_ path (and for all its children too)
                else:
                    token.features[Part._FEAT_DEPTH_KEY] = current_depth
                    token_children = [child for (child, _) in token.features['dependency_to']]
                    next_depth_tokens += token_children

            Part._recursive_compute_tokens_depth(current_depth + 1, next_depth_tokens)


    def set_entities_head_tokens(self):
        """
        **Depends on** __class__::compute_tokens_depth

        A head token of an entity is an arbitrary definition of ours.
        It roughly means "the most important token of an entity's token list".

        Shrikant, essentially, defined this heuristically setting the head as the "root"
        of the entity. Meaning, the token that is closest to an actual root of the dependency graph.

        Shrikant code (Sentence::calculateHeadScores) also considered "important" dependencies
        as to only follow those to calculate "closedness" to the root (depth), and further,
        also penalized punctuation tokens.

        Ashish supposedly implemented the same idea in the deprecated `calculate_token_scores`.

        YET, in my view, BOTH Shrikant's and Ashish's implementations and logics were wrong
        (if we follow the idea of their methods' descriptions).


        In this new implementation (author @juanmirocks), we calculate the "score" of tokens
        with the new function `compute_tokens_depth` and we consider as head of an entity:

        *   the token that has the least depth. That is, e.g., depth 2 "wins" over depth 3.
        *   Further, all-punctuation tokens will never be selected as entity head tokens
            (unless they are the sole token of the entity; this should never happen anyway
            and we assert it appropriately; maybe special unicode characters can slip through, e.g. delta).
        *   Further, upon tokens with same depth, the token that is a Noun wins (Token::is_POS_Noun)
        *   Finally, if still multiple token candidates remain, the first is arbitrarily selected.

        """

        # The following code can be made more efficient by iterating only once over the tokens lists
        # The code will become more verbose, thoughss

        for e in chain(self.annotations, self.predicted_annotations):

            tokens = e.tokens

            # Filter out punctuation tokens
            tokens = list(filter(lambda t: not t.features['is_punct'], tokens))
            assert len(tokens) >= 1, (e, " --> ", tokens)

            # Leave only minum depth tokens
            minimum_depth = min((t.features[Part._FEAT_DEPTH_KEY] for t in tokens))
            tokens = [t for t in tokens if t.features[Part._FEAT_DEPTH_KEY] == minimum_depth]

            # Leave only nouns
            nn_tokens = [t for t in tokens if t.is_POS_Noun()]

            if len(nn_tokens) == 0:
                # print_debug("No Noun in the entity tokens", (e, e.tokens))
                e.head_token = tokens[0]

            else:
                e.head_token = nn_tokens[0]

                # if len(nn_tokens) > 1:
                #     print_debug("Same score for entity head tokens", (e.text, "head: ", e.head_token, "nn tokens: ", nn_tokens))


    def calculate_token_scores(self):
        """
        calculate score for each entity based on a simple heuristic of which
        token is closest to the root based on the dependency tree.

        @
        """
        warnings.warn('Use `compute_tokens_depth` instead', DeprecationWarning)

        not_tokens = []
        # important_dependencies = [
        #     'det', 'amod', 'appos', 'npadvmod', 'compound',
        #     'dep', 'with', 'nsubjpass', 'nsubj', 'neg', 'prep', 'num', 'punct'
        # ]
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
                    # dep_type = token.features['dependency_from'][1]

                    # if dep_type in important_dependencies:
                    if True:
                        if dep_from.features['score'] <= dep_to.features['score']:
                            dep_from.features['score'] = dep_to.features['score'] + 1
                            done = False
                counter += 1
                if counter > 20:
                    break


    def set_head_tokens(self):
        """
        set head token for each entity based on the scores for each token
        """
        warnings.warn('Use `set_entities_head_tokens` instead', DeprecationWarning)

        for token in self.tokens:
            if token.features['score'] is None:
                token.features['score'] = 1

        for entity in chain(self.annotations, self.predicted_annotations):
            if len(entity.tokens) == 1:
                entity.head_token = entity.tokens[0]
            else:
                entity.head_token = max(entity.tokens, key=lambda token: token.features['score'])


    def map_relations(self, use_predicted, relation_type, entity_map_fun, relations_search_space=None, part_mapped_relations=None):
        """
        Map all Parts's relations to dictionary of:
        {unique mapped strings --> (list of tuples: (relation with same map key, sentence distance between the related entities))}

        Create a set of the document's relations based on the map function of the relation themselves and the given map
        function for their entities. Relations end up being represented as strings in the set.

        If relations_search_space is None, return the map of all parts' relations. Otherwise, return the map
        of only those part's relations that are included in `relations_search_space` (a set or a list)

        Return: set of strings that represent unique relationsihps
        """

        if part_mapped_relations is None:
            part_mapped_relations = {}

        part_relations = self.predicted_relations if use_predicted else self.relations

        for r in part_relations:
            if r.class_id == relation_type and (relations_search_space is None or r in relations_search_space):
                mapkey = r.map(entity_map_fun)

                if mapkey is not None:
                    equivalent = part_mapped_relations.get(mapkey, [])
                    entities_sentence_distance = r.get_sentence_distance_between_entities(self)
                    equivalent.append((r, entities_sentence_distance))
                    part_mapped_relations[mapkey] = equivalent

        return part_mapped_relations


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
               '-Predicted relations-\n{pred_relations}'.format(
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

    Asserted:
        The entities are assumed to be sorted, that is:
            * their parts are sorted ( <= )  # can be same part
            * (if same parts), their sentences are sorted ( <= )  # can be same sentence
            * (if same parts), their start offsets are sorted ( < )  # < for they must be different entities/tokens

    Note:
        The same_ (part or sentence_id) are sugar fields for convenience.
        Their use is discouraged and ideally the library would throw a warning when used.

        The best best solution would be to be able to retrieve the part and sentence_id from the entities directly
    """

    def __init__(self, relation_type, entity1, entity2, e1_part, e2_part, e1_sentence_id, e2_sentence_id):
        assert e1_part.sentences[0][0].start <= e2_part.sentences[0][0].start, ("Parts must be sorted", e1_part, e2_part)
        assert e1_part != e2_part or e1_sentence_id <= e2_sentence_id, ("Sentences must be sorted", e1_sentence_id, e2_sentence_id)
        assert e1_part != e2_part or entity1.offset < entity2.offset, ("Entities must be sorted", e1_sentence_id, e2_sentence_id, entity1, entity2)

        #

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

        self.__combined_sentence = None
        """
        Private pre-computed field. See method `get_combined_sentence`
        """

        self.features = {}
        """
        A dictionary of features for the edge.
        Each feature is represented as a key value pair:
            The key is an integer that corresponds to a feature key in the dataset's feature_set (i.e. a value in this dictionary)
            The value is the feature's value in this edge
        """

        self.features_vector = None
        """
        None if not set, otherwise scipy sparse vector-like array with finally-encoded features
        """

        self.real_target = None
        """real class of the edge -- ASSUMED to be in {-1, +1} or None when not defined"""

        self.pred_target = None
        """predicted class of the edge -- ASSUMED to be in {-1, +1} or None when not defined"""

        self.initial_instance_index = None
        """row index of this edge in the matrix X of instances gathered initially for all data (training + test)"""


    def __repr__(self):
        return 'Edge between "{0}" and "{1}" of the type "{2}".'.format(self.entity1.text, self.entity2.text, self.relation_type)


    def has_same_sentences(self):
        return self.e1_sentence_id == self.e2_sentence_id


    def get_any_entities_in_sentences(self, predicted):
        assert self.same_part

        s1 = self.same_part.get_any_entities_in_sentence(self.e1_sentence_id, predicted)

        if not self.has_same_sentences():
            s2 = self.same_part.get_any_entities_in_sentence(self.e2_sentence_id, predicted)

            for key, vals in s2.items():
                updated = s1.get(key, [])
                updated += vals
                s1[key] = updated

        return s1


    def get_any_entities_between_entities(self, predicted):
        assert self.same_part

        ret = {}
        for e_class_id, entities in self.get_any_entities_in_sentences(predicted).items():
            keep = []
            for entity in entities:
                if self.entity1.end_offset() <= entity.offset < self.entity2.offset:
                    keep.append(entity)
            ret[e_class_id] = keep

        return ret


    def get_potential_relation(self):
        """
        Get the potential relation represented by this edge.
        """
        ret = Relation(self.relation_type, self.entity1, self.entity2)
        assert ret.bidirectional, "Code tested only for bidirectional relations"
        return ret


    def get_relation_if_is_real(self):
        """
        If this edge represents a _real_ relation, return this -- Otherwise return None
        """
        if self.real_target == +1 or self.is_relation():
            return self.get_potential_relation()
        else:
            return None


    def get_potential_relation_if_is_predicted(self):
        """
        If this edge is _predicted_ to be relation, return its representation -- Otherwise return None

        Note: likely you do not need this -- Sugar function just for completeness.
        """
        if self.pred_target == +1:
            return self.get_potential_relation()
        else:
            return None


    def is_relation(self):
        """
        check if the edge is present in part.relations.
        :rtype: bool
        """
        potential_edge_relation = self.get_potential_relation()

        relations = self.same_part.relations if self.e1_part == self.e2_part else chain(self.e1_part.relations, self.e2_part.relations)

        return potential_edge_relation in relations


    def get_sentences_pair(self):
        """
        Get tuple of corresponding edge's two entities' sentences.
        The sentences are represented as list of Token's.
        """

        assert self.e1_sentence_id != self.e2_sentence_id or self.same_sentence_id

        sent1 = self.e1_part.sentences[self.e1_sentence_id]
        sent2 = self.e2_part.sentences[self.e2_sentence_id]

        return (sent1, sent2)


    def get_entity2_offset(self, original_offset=0):
        if self.has_same_sentences():
            return 0 + original_offset
        else:
            # If they are not in the same sentence, the sentences are combined and we need to add the extra offset of sentence 1
            sent1 = self.e1_part.sentences[self.e1_sentence_id]
            return len(sent1) + original_offset


    def get_combined_sentence(self, recreate_user_dependencies=True):
        # Currently we do not reuse the internal field becaus e of caveats with the user dependencies and temporal ids

        if self.has_same_sentences():
            self.__combined_sentence = self.e1_part.sentences[self.e1_sentence_id]
            if recreate_user_dependencies:
                for t in self.__combined_sentence:
                    # Same as tmp_id, these features get recreated with each call
                    t.features['user_dependency_to'] = []
                    t.features['user_dependency_from'] = []
        else:
            self.__combined_sentence = __class__._combine_sentences(self, *self.get_sentences_pair())

        for index, t in enumerate(self.__combined_sentence):
            t.features['tmp_id'] = index  # The tmp_id will be rewritten each time an edge calls this method; beware

        return self.__combined_sentence

###

    @staticmethod
    def _combine_sentences(edge, sentence1, sentence2, recreate_user_dependencies=True):
        """
        Combine two simple simple normal sentences into a "chained" sentence with
        dependecies and paths created as necessary for the DS model.

        `createCombinedSentence` re-implementation of Shrikant's (java) into Python.

        Each sentence is a list of Tokens as defined in class Part (nalaf: data.py).

        The sentences are assumed, but not asserted, to be different and sorted:
        sentence1 must be before sentence2.
        """

        combined_sentence = sentence1 + sentence2

        if recreate_user_dependencies:
            for t in combined_sentence:
                # Same as tmp_id, these features get recreated with each call
                t.features['user_dependency_to'] = []
                t.features['user_dependency_from'] = []

        combined_sentence = __class__._add_extra_links(edge, combined_sentence, sentence1, sentence2)

        return combined_sentence


    @staticmethod
    def _add_extra_links(edge, combined_sentence, sentence1, sentence2):
        """
        `addExtraLinks` re-implementation of Shrikant's (java) into Python.

        Some comments and commented-out code exactly as original java code.
        """

        __class__._addRootLinks(edge, combined_sentence, sentence1, sentence2)

        __class__._addWordSimilarityLinks(edge, combined_sentence, sentence1, sentence2)

        # TODO add ?
        # TODO would be better to not use the constants PRO_ID (protRef) and LOC_ID (locRef) (below) here -- It's hardcoded
        # _addEntityLinks(edge, combined_sentence, sentence1, sentence2, PRO_ID, ['protein'], "protRef")

        # TODO add ?
        # Just as we added the links from "protein" to actual protein entities
        # add the links from "location"/"localization" to location entity
        # _addEntityLinks(edge, combined_sentence, sentence1, sentence2, LOC_ID, ['location', 'localiz', 'com.ment'], "locRef")

        # TODO
        # addProteinFamilyLinks(combSentence, tokenOffset);

        # TODO
        # addShortFormLinks(combSentence, prevSentence, currSentence)

        return combined_sentence


    @staticmethod
    def _addRootLinks(edge, combined_sentence, sentence1, sentence2):
        """
        link roots of both the sentences

        `addRootLinks` re-implementation of Shrikant's (java) into Python.


        *IMPORTANT*:

        * Shrikant/Java/CoreNLP code had one single root for every sentence
        * Python/spaCy sentences can have more than 1 root
        * --> Therefore, we create a product of links of all the roots
        * --> see: (https://github.com/juanmirocks/LocText/issues/6#issue-177139892)


        Dependency directions:

        sentence1 -> sentence2
        sentence2 <- sentence1
        """
        from itertools import product

        for (s1_root, s2_root) in product(Part.get_sentence_roots(sentence1), Part.get_sentence_roots(sentence2)):

            s1_root.features['user_dependency_to'].append((s2_root, "rootDepForward"))
            s1_root.features['user_dependency_from'].append((s2_root, "rootDepBackward"))

            s2_root.features['user_dependency_from'].append((s1_root, "rootDepForward"))
            s2_root.features['user_dependency_to'].append((s1_root, "rootDepBackward"))


    @staticmethod
    def _addWordSimilarityLinks(edge, combined_sentence, sentence1, sentence2):
        """
        For now:

        * Add links between the product of tokens that are Noun and have same lemma.
        """
        from itertools import product

        for s1_token, s2_token in product(sentence1, sentence2):

            # Maybe just Noun's conditions? maybe only entities condition? Both?
            if (s1_token.is_POS_Noun() and s2_token.is_POS_Noun()) or \
                (s1_token.get_entity(edge.same_part, True, True) is not None and s2_token.get_entity(edge.same_part, True, True) is not None):

                if s1_token.features['lemma'] == s2_token.features['lemma']:
                    s1_token.features['user_dependency_to'].append((s2_token, "same_lemma"))
                    s2_token.features['user_dependency_from'].append((s1_token, "same_lemma"))


    @staticmethod
    def _addEntityLinks(edge, combined_sentence, sentence1, sentence2, class_id, key_words, dependency_type):
        """
        `addProteinLinks` and `addLocationLinks` re-implementation of Shrikant's (java) into Python.
        """

        assert edge.same_part

        def _do_one_direction(sent_a, sent_b):

            sent_a_contains_entity = edge.entity1.class_id == class_id
            if not sent_a_contains_entity:
                sent_a_tokens_that_match_key_words = (t for t in sent_a if any(kw in t.word.lower() for kw in key_words))

                for sent_a_token in sent_a_tokens_that_match_key_words:

                    for sent_b_token in sent_b:

                        sent_b_token_in_entity = sent_b_token.get_entity(edge.same_part)  # TODO use_pred ?
                        if sent_b_token_in_entity is not None and sent_b_token_in_entity.class_id == class_id:

                            sent_a_token.features['user_dependency_to'].append((sent_b_token, dependency_type))
                            sent_b_token.features['user_dependency_from'].append((sent_a_token, dependency_type))

        # In combination: do both directions
        _do_one_direction(sentence1, sentence2)
        _do_one_direction(sentence2, sentence1)


class Token:
    """
    Represent a token - the smallest unit on which we perform operations.
    Usually one token represent one word from the document.

    :type word: str
    :type start: int
    :type end: int
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


    def is_POS_Noun(self):
        """ matches NN, NNS, NNP, NNPS : https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html"""
        return "NN" == self.features['pos'][0:2]


    def is_POS_Verb(self):
        return "V" == self.features['pos'][0]


    def get_entity(self, part, use_gold, use_pred):
        """
        if the token is part of an entity, return the entity else return None
        :param part: an object of type Part in which to search for the entity.
        :type part: nalaf.structures.data.Part
        :return nalaf.structures.data.Entity or None
        """
        entities = chain(
            part.annotations if use_gold else [],
            part.predicted_annotations if use_pred else []
        )

        for entity in entities:
            if entity.offset <= self.start < entity.end_offset() or entity.offset < self.end <= entity.end_offset():
                return entity

        return None


class FeatureDictionary(dict):
    """
    Extension of the built in dictionary with:

    1) the added constraint that keys (feature names) cannot be updated.
       It raises an exception if you try to add a key that exists already.

    2) Possibility to lock the dictionary to prohibit any change (i.e. make the object immutable)

    3) TODO: MAY BE DISCARDED
       If the key (feature name) doesn't end with "[number]" appends "[0]" to it.
       This is used to identify the position in the window for the feature.
    """

    def __init__(self, is_locked=False):
        self.is_locked = is_locked

    def __setitem__(self, key, value):
        if key in self:
            raise KeyError('feature name "{}" already exists'.format(key))
        else:
            # TODO this would be better written in the (entities) FeatureGenerator
            if not re.search('\[-?[0-9]+\]$', key):
                key += '[0]'
            dict.__setitem__(self, key, value)


class Entity:
    """
    Represents a single entity, i.e. a text span, which names/refers to a defined concept.

    :type class_id: str
    :type offset: int
    :type text: str
    :type subclass: int
    :type confidence: float
    :type norms: dict
    :type normalized_text: str
    :type tokens: list[nalaf.structures.data.Token]
    :type head_token: nalaf.structures.data.Token
    """

    def __init__(self, class_id, offset, text, confidence=1, norms=None):
        self.class_id = class_id
        """the id of the class of entity (concept) that is annotated"""

        self.offset = offset
        """the offset marking the beginning of the annotation in regards to the Part this annotation is attached to."""

        self.text = text
        """the text span of the annotation"""

        self.subclass = False
        # TODO likely, we should not allow subclasses that are not string in the first place -- to the very least, the default should be None
        # Explaination in commit: 3983e4c5449788e62e81b39b65fc7780b6c71852
        """
        int flag used to further subdivide classes based on some criteria
        for example for mutations (MUT_CLASS_ID): 0=standard, 1=natural language, 2=semi standard
        """

        self.confidence = confidence
        """aggregated mention level confidence from the confidence of the tokens based on some aggregation function"""

        self.norms = {} if norms is None else norms
        """
        Dictionary of normalization ids if normalization (i.e. entity disambiguation) was performed.

        A same entity can be linked to different databases (through an unique id).
        And even within the same database, the entity could have different ids (represented as a comma-separated string)

        Example, an entity linked to database referred as `uac` could have the ids `Q9P2K8` and `P15442`:

        {'n_7': 'Q9P2K8,P15442,Q9LX30,Q9FIB4'}
        """

        self.normalized_text = ''
        """(OFTEN NOT USED) the normalized text for the annotation if normalization was performed"""

        self.tokens = []
        """
        The tokens of the entity.

        YOU MUST CALL BEFORE: the entity's part percolate_tokens_to_entities()

        TODO Note that tokens are already within sentences. You should use those by default.
        This list of tokens may be deleted. See: https://github.com/Rostlab/nalaf/issues/167
        """

        self.sentence = None
        """
        The whole sentence of tokens this entity belongs to, if set.

        YOU MUST CALL BEFORE: the entity's part percolate_tokens_to_entities()
        """

        self.part = None
        """
        The whole part this entity belongs to, if set.

        YOU MUST CALL BEFORE: the entity's part percolate_tokens_to_entities()
        """

        self.head_token = None
        """the head token for the entity. Note: this is not necessarily the first token, just the head of the entity as declared by parsing (see parsers.py)"""

        self.features = {}
        """
        User-defined object with dictionary of features (names to values)
        """


    equality_operator = 'exact'
    """
    determines when we consider two entities to be equal
    can be "exact" or "overlapping" or "exact_or_overlapping"
    """

    def end_offset(self):
        return self.offset + len(self.text)


    def __repr__(self):
        subclass_str = (" (" + str(self.subclass) + ")") if self.subclass else ""

        if self.norms:
            norm_str = ', norms: {}'.format(self.norms)
        else:
            norm_str = ''

        return 'Entity(class_id: {}{}, offset: {}, ' \
               'text: {}{})'.format(self.class_id, subclass_str, self.offset, self.text, norm_str)


    def __eq__(self, that):
        # consider them a match only if class_id matches
        # TODO implement test case for edge cases in overlap and exact
        if self.class_id == that.class_id:
            exact = self.offset == that.offset and self.text == that.text
            overlap = self.offset < that.end_offset() and self.end_offset() > that.offset

            if self.equality_operator == 'exact':
                return exact
            elif self.equality_operator == 'overlapping':
                # overlapping means only the case where we have an actual overlap and not exact match
                return not exact and overlap
            elif self.equality_operator == 'exact_or_overlapping':
                # overlap includes the exact case so just return that
                return overlap
            else:
                raise ValueError('that must be "exact" or "overlapping" or "exact_or_overlapping"')
        else:
            return False


    def prev_tokens(self, sentence, n, include_ent_first_token=False, mk_reversed=False):
        self_first = self.tokens[0].features['id']
        right_index = self_first + 1 if include_ent_first_token else self_first
        left_index = max(0, self_first - n)
        ret = sentence[left_index:right_index]
        return list(reversed(ret)) if mk_reversed else ret


    def next_tokens(self, sentence, n, include_ent_last_token=False):
        self_last = self.tokens[-1].features['id']
        left_index = self_last if include_ent_last_token else self_last + 1
        right_index = self_last + 1 + n
        return sentence[left_index:right_index]


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

    def __init__(self, relation_type, entity1, entity2, bidirectional=True):
        self.class_id = relation_type

        assert entity1 is not None and entity2 is not None, "Some of the entities are None"

        self.entity1 = entity1
        self.entity2 = entity2

        self.bidirectional = bidirectional


    def __repr__(self):
        return 'Relation(class_id:"{self.class_id}": e1:"{self.entity1}"   <--->   e2:"{self.entity2}")'.format(self=self)


    def map(self, entity_map_fun, prefix_with_rel_type=True):
        e1_string = entity_map_fun(self.entity1)
        e2_string = entity_map_fun(self.entity2)

        if e1_string is None or e2_string is None:
            return None

        else:
            if (self.bidirectional and self.entity2.class_id <= self.entity1.class_id):
                entities = [e2_string, e1_string]
            else:
                entities = [e1_string, e2_string]

            if prefix_with_rel_type:
                items = [self.class_id, *entities]
            else:
                items = entities

            return '|'.join(items)


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
            if ann.offset == self.entity2.offset and ann.text == self.entity2.text:
                second = True
            if first and second:
                return True

        return False

    def get_relation_without_offset(self):
        """:return string with entity1 and entity2 separated by relation type"""
        return (self.entity1.text, self.class_id, self.entity2.text)


    def __eq__(self, other):
        """
        consider two relations equal if and only if all their parameters match
        :type other: nalaf.structures.data.Relation
        :return bool:
        """

        if other is not None:
            return (self.class_id == other.class_id and
                    self.bidirectional == other.bidirectional and
                    (self.entity1 == other.entity1 and self.entity2 == other.entity2 or
                        self.bidirectional and self.entity1 == other.entity2 and self.entity2 == other.entity1))

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


    def get_sentence_distance_between_entities(self, same_part):
        index1 = same_part.get_sentence_index_for_annotation(self.entity1)
        index2 = same_part.get_sentence_index_for_annotation(self.entity2)
        distance = abs(index1 - index2)
        return distance
