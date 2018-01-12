import abc
from nalaf.structures.data import Entity
from nalaf import print_verbose, print_debug
from collections import namedtuple
import random
import math
import uuid
from collections import Counter
import re


class Evaluation:

    Computation = namedtuple('Computation', ['precision', 'recall', 'f_measure'])

    def __init__(self, label, tp, fp, fn, fp_ov=0, fn_ov=0):
        # TODO
        import warnings
        warnings.warn('`Evaluation` will be removed to only leave EvaluationWithStandardError (complete superset of functionality)')

        self.label = str(label)
        self.tp = tp
        self.fp = fp
        self.fn = fn
        self.fp_ov = fp_ov
        self.fn_ov = fn_ov

    def compute(self, strictness):
        """
        strictness:
        Determines whether a text spans matches and how we count that match, 3 possible values:
            * 'exact' count as:
                1 ONLY when we have exact match: (startA = startB and endA = endB)
            * 'overlapping' count as:
                1 when we have exact match
                1 when we have overlapping match
            * 'half_overlapping' count as:
                1 when we have exact match
                0.5 when we have overlapping match
        """

        if strictness == 'exact':
            precision = self._safe_div(self.tp, self.tp + self.fp)
            recall = self._safe_div(self.tp, self.tp + self.fn)

        elif strictness == 'overlapping':
            fp = self.fp - self.fp_ov
            fn = self.fn - self.fn_ov
            tp = self.tp + self.fp_ov + self.fn_ov

            precision = self._safe_div(tp, tp + fp)
            recall = self._safe_div(tp, tp + fn)

        elif strictness == 'half_overlapping':
            fp = self.fp - self.fp_ov
            fn = self.fn - self.fn_ov

            precision = self._safe_div(self.tp + (self.fp_ov + self.fn_ov) / 2, self.tp + self.fp_ov + self.fn_ov + fp)
            recall = self._safe_div(self.tp + (self.fp_ov + self.fn_ov) / 2, self.tp + self.fp_ov + self.fn_ov + fn)

        else:
            raise ValueError('strictness must be "exact" or "overlapping" or "half_overlapping"')

        f_measure = 2 * self._safe_div(precision * recall, precision + recall)

        return Evaluation.Computation(precision, recall, f_measure)

    def __str__(self):
        return '\n'.join([self.format_header(), self.format_row()])

    def format_header(self, strictnesses=None):
        strictnesses = ['exact', 'overlapping'] if strictnesses is None else strictnesses

        header = ['# class', 'tp', 'fp', 'fn', 'fp_ov', 'fn_ov']
        for _ in strictnesses:
            header += ['match', 'P', 'R', 'F']
        return '\t'.join(header)

    def _format_counts_list(self):
        ret = [self.tp, self.fp, self.fn, self.fp_ov, self.fn_ov]
        return [str(c) for c in ret]

    def format_row(self, strictnesses=None):
        strictnesses = ['exact', 'overlapping'] if strictnesses is None else strictnesses

        cols = [self.label] + self._format_counts_list()
        for strictness in strictnesses:
            cols += [strictness[0]]  # first character
            cols += self.format_computation(self.compute(strictness))
        return '\t'.join(cols)

    def format_computation(self, c):
        complist = [c.precision, c.recall, c.f_measure]
        return ["{:6.4f}".format(n) for n in complist]

    @staticmethod
    def _safe_div(nominator, denominator):
        try:
            return nominator / denominator
        except ZeroDivisionError:
            return 0.0  # arbitrary; or float('NaN')


class EvaluationWithStandardError:

    Computation = namedtuple('Computation',
                             ['precision', 'precision_SE', 'recall', 'recall_SE', 'f_measure', 'f_measure_SE'])

    def __init__(self, label, dic_counts, n=1000, p=0.15, mode='macro', precomputed_SEs=None):
        self.label = str(label)
        self.dic_counts = dic_counts
        self.n = n
        self.p = p
        assert mode == 'macro', "`micro` mode is not implemented yet"
        self.mode = mode
        self.precomputed_SEs = precomputed_SEs

        self.keys = dic_counts.keys()
        self.keys_len = len(self.keys)
        self._mean_eval = Evaluation(
            str(self.label),
            self._get('tp'), self._get('fp'), self._get('fn'), self._get('fp_ov'), self._get('fn_ov'))

        self.tp = self._mean_eval.tp
        self.fp = self._mean_eval.fp
        self.fn = self._mean_eval.fn
        self.fp_ov = self._mean_eval.fp_ov
        self.fn_ov = self._mean_eval.fn_ov

    def _get(self, count, keys=None):
        if keys is None:
            keys = self.keys

        return sum([counts.get(count, 0) for key, counts in self.dic_counts.items() if key in keys])

    def _compute_SE(self, mean, array, multiply_small_values=4):
        cleaned = [x for x in array if not math.isnan(x)]
        n = len(cleaned)
        ret = Evaluation._safe_div(math.sqrt(sum((x - mean) ** 2 for x in cleaned) / (n - 1)), math.sqrt(n))
        if (ret <= 0.00001):
            ret *= multiply_small_values
        return ret

    def compute(self, strictness, precomputed_SE=None):
        means = self._mean_eval.compute(strictness)

        if precomputed_SE is None:
            samples = []
            for _ in range(self.n):
                random_keys = random.sample(self.keys, round(self.keys_len * self.p))  # without replacement
                sample = Evaluation(str(self.label),
                                    self._get('tp', random_keys),
                                    self._get('fp', random_keys),
                                    self._get('fn', random_keys),
                                    self._get('fp_ov', random_keys),
                                    self._get('fn_ov', random_keys))

                samples.append(sample.compute(strictness))

            p_SE = self._compute_SE(means.precision, [sample.precision for sample in samples])
            r_SE = self._compute_SE(means.recall, [sample.recall for sample in samples])
            f_SE = self._compute_SE(means.f_measure, [sample.f_measure for sample in samples])

        else:
            p_SE = precomputed_SE['precision_SE']
            r_SE = precomputed_SE['recall_SE']
            f_SE = precomputed_SE['f_measure_SE']

        return EvaluationWithStandardError.Computation(
            means.precision, p_SE,
            means.recall, r_SE,
            means.f_measure, f_SE)


    def __str__(self):
        return '\n'.join([self.format_header(), self.format_row()])


    def format_header(self, strictnesses=None):
        return self.format_header_simple(strictnesses)

    def format_row(self, strictnesses=None):
        return self.format_row_simple(strictnesses)

    def format_header_complete(self, strictnesses=None):
        strictnesses = ['exact', 'overlapping'] if strictnesses is None else strictnesses

        header = ['# class', 'tp', 'fp', 'fn', 'fp_ov', 'fn_ov']
        for _ in strictnesses:
            header += ['match', 'P', 'P_SE', 'R', 'R_SE', 'F', 'F_SE']
        return '\t'.join(header)

    def format_row_complete(self, strictnesses=None):
        strictnesses = ['exact', 'overlapping'] if strictnesses is None else strictnesses

        cols = [self.label] + self._mean_eval._format_counts_list()
        for strictness in strictnesses:
            cols += [strictness[0]]  # first character
            precomputed_SE = self.precomputed_SEs[strictness] if self.precomputed_SEs else None
            cols += self.format_computation_simple(self.compute(strictness, precomputed_SE))
        return '\t'.join(cols)

    def format_header_simple(self, strictnesses=None):
        strictnesses = ['exact', 'overlapping'] if strictnesses is None else strictnesses

        header = ['# class', 'tp', 'fp', 'fn', 'fp_ov', 'fn_ov']
        for strictness in strictnesses:
            s = strictness[0].lower() + "|"  # first letter
            header += [s + c for c in ['P', 'R', 'F', 'F_SE']]
        return '\t'.join(header)

    def format_row_simple(self, strictnesses=None):
        strictnesses = ['exact', 'overlapping'] if strictnesses is None else strictnesses

        cols = [self.label] + self._mean_eval._format_counts_list()
        for strictness in strictnesses:
            precomputed_SE = self.precomputed_SEs[strictness] if self.precomputed_SEs else None
            cols += self.format_computation_simple(self.compute(strictness, precomputed_SE))
        return '\t'.join(cols)

    def _num_leading_zeros(self, num_str):
        ret = 0
        for digit in num_str:
            if not digit.isdigit():  # digital symbol: . or ,
                continue
            if digit == '0':
                ret += 1
            else:
                return ret
        return ret

    def format_computation_complete(self, c):
        complist = [c.precision, c.precision_SE, c.recall, c.recall_SE, c.f_measure, c.f_measure_SE]
        return ["{:6.4f}".format(n) for n in complist]

    def format_computation_simple(self, c):
        complist = [c.precision, c.recall, c.f_measure, c.f_measure_SE]
        return ["{:6.4f}".format(n) for n in complist]

    def format_computation_complete_removing_noise(self, c):
        """
        Caveat: it does not work wie SE values that equal NaN
        """

        ses = [c.precision_SE, c.recall_SE, c.f_measure_SE]
        ses = ["{:5.3f}".format(x * 100) for x in ses]
        ses_zeros = [self._num_leading_zeros(x) for x in ses]
        ses_zeros = [(1 if x == 0 else x) for x in ses_zeros]
        ses_ints = [x.index('.') for x in ses]
        # number of digits depends on respective SE
        ses_reduced = [se[:(integer+1+zeros)] for integer, zeros, se in zip(ses_ints, ses_zeros, ses)]
        ses_formats = [("{:6."+str(x)+"f}") for x in ses_zeros]
        comps = [sef.format(x * 100) for sef, x in zip(ses_formats, [c.precision, c.recall, c.f_measure])]
        comps = [c[:min(6, len(c))] for c in comps]

        return [item for pair in zip(comps, ses_reduced) for item in pair]


class Evaluations:

    def __init__(self):
        self.classes = {}

    def __call__(self, clazz):
        return self.classes[str(clazz)]

    def add(self, evaluation):
        self.classes[str(evaluation.label)] = evaluation

    def __str__(self):
        return self.format()

    def format(self, strictnesses=None, add_SE='macro'):
        strictnesses = ['exact', 'overlapping'] if strictnesses is None else strictnesses

        assert(len(self.classes) >= 1)
        rows = [next(iter(self.classes.values())).format_header(strictnesses)]
        for clazz in sorted(self.classes.keys()):
            evaluation = self.classes[clazz]
            rows += [evaluation.format_row(strictnesses)]
        return '\n'.join(rows)


    def __iter__(self):
        return self.classes.__iter__()


    @staticmethod
    def merge(evaluations_itr, are_disjoint_evaluations=True):
        """
        Common use case: combine cross-validation evaluations
        The single evaluations are assumed to be of type EvaluationWithStandardError
        """

        labels = {}

        for evaluations in evaluations_itr:
            for evaluation_label, evaluation in evaluations.classes.items():
                dic_counts = labels.get(evaluation.label, {})

                if are_disjoint_evaluations:
                    joint_size = len(set(dic_counts.keys()) & set(evaluation.dic_counts))
                    assert joint_size == 0, "The evaluations are assumed to be disjoint, yet they are not. Joint size: " + str(joint_size)

                    dic_counts.update(evaluation.dic_counts)

                else:
                    for docid, eval_docid_counts in evaluation.dic_counts.items():

                        total_docid_counts = dic_counts.get(docid, None)

                        if total_docid_counts:
                            counter = Counter()
                            for d in [total_docid_counts, eval_docid_counts]:
                                counter.update(d)

                            dic_counts[docid] = dict(counter)

                        else:
                            dic_counts[docid] = eval_docid_counts

                labels[evaluation.label] = dic_counts

        ret = Evaluations()

        for label, dic_counts in labels.items():
            ret.add(EvaluationWithStandardError(label, dic_counts))

        return ret


    @staticmethod
    def cross_validate(annotator_gen_fun, corpus, evaluator, k_num_folds, use_validation_set=True):
        merged_evaluations = []

        print_debug("Cross-Validation")
        for training_set, evaluation_set in corpus.cv_kfold_splits(k_num_folds, validation_set=use_validation_set):

            annotator_apply = annotator_gen_fun(training_set)
            annotator_apply(evaluation_set)

            r = evaluator.evaluate(evaluation_set)
            print_debug(r)
            merged_evaluations.append(r)

        ret = Evaluations.merge(merged_evaluations)
        print_debug("\n" + str(ret) + "\n")

        return ret


class Evaluator:
    """
    Calculates precision, recall and subsequently F1 measure based on the original and the predicted mention
    to evaluate the performance of a model.

    Different implementations are possible based on the level of consideration such as:
        * Token level
        * Mention level
        * etc...
    """

    @abc.abstractmethod
    def evaluate(self, dataset):
        """
        :type dataset: nalaf.structures.data.Dataset
        :returns Evaluations
        """
        return


class MentionLevelEvaluator(Evaluator):

    TOTAL_LABEL = "TOTAL"

    """
    Implements mention level performance evaluation. That means it compares if the predicted text spans match
    the original annotated text spans.

    Whether a text spans matches and how we count that match is determined
    by the value of the parameter 'strictness'.
    """

    def __init__(self, subclass_analysis=False):
        # TODO
        import warnings
        warnings.warn('`MentionLevelEvaluator` is deprecated in favor of `EntityEvaluator`')

        self.subclass_analysis = subclass_analysis
        """
        Whether to report the performance for each subclass separately
        """


    def evaluate(self, dataset):
        """
        :type dataset: nalaf.structures.data.Dataset
        :returns (tp, fp, fn, tp_overlapping, precision, recall, f_measure): (int, int, int, int, float, float, float)

        Calculates precision, recall and subsequently F1 measure, defined as:
            * precision: number of correctly predicted items as a percentage of the total number of predicted items
                len(predicted items that are also real)/len(predicted)
                or in other words tp / tp + fp
            * recall: number of correctly predicted items as a percentage of the total number of correct items
                len(real items that are also predicted)/len(real)
                or in other words tp / tp + fn
            * possibly considers overlapping matches as well
        """

        TOTAL = MentionLevelEvaluator.TOTAL_LABEL
        labels = [TOTAL]

        def labelize(e):
            """
            Use this to represent an entity subclass as string and, if this is None or False (but not 0!), represent the entity with its class_id

            Convert to subclasses / classes ids to avoid the misstep of comparing possible subclass '0' with False, which in python breaks the universe
            --> info: https://twitter.com/juanmirocks/status/802209750612054016
            """
            return str(e.subclass) if str(e.subclass) not in ['None', 'False'] else str(e.class_id)

        if self.subclass_analysis:
            # find all possible subclasses or otherwise full classes

            subclasses = set(labelize(e) for e in dataset.entities())
            subclasses.update(set(labelize(e) for e in dataset.predicted_entities()))

            for x in subclasses:
                labels.append(x)

        docids = dataset.documents.keys()
        subcounts = ['tp', 'fp', 'fn', 'fp_ov', 'fn_ov']
        counts = {label: {docid: dict.fromkeys(subcounts, 0) for docid in docids} for label in labels}

        for docid, doc in dataset.documents.items():
            for partid, part in doc.parts.items():

                overlap_real = {label: [] for label in labels}
                overlap_predicted = {label: [] for label in labels}

                Entity.equality_operator = 'overlapping'
                for ann_a in part.annotations:
                    for ann_b in part.predicted_annotations:
                        if ann_a == ann_b:  # equal according according to exclusive overlapping eq (not exact)
                            overlap_real[TOTAL].append(ann_a)
                            overlap_predicted[TOTAL].append(ann_b)

                            if self.subclass_analysis:
                                if labelize(ann_a) != labelize(ann_b):
                                    print_debug('overlapping subclasses do not match', ann_a.subclass, ann_b.subclass)
                                    ann_b.subclass = ann_a.subclass

                                overlap_real[labelize(ann_a)].append(ann_a)
                                overlap_predicted[labelize(ann_b)].append(ann_b)

                Entity.equality_operator = 'exact'
                for ann in part.predicted_annotations:
                    if ann in part.annotations:
                        counts[TOTAL][docid]['tp'] += 1
                        print_verbose("    ", docid, ": TRUE POSITVE", ann)

                        if self.subclass_analysis:
                            counts[labelize(ann)][docid]['tp'] += 1

                    else:
                        counts[TOTAL][docid]['fp'] += 1

                        if ann in overlap_predicted[TOTAL]:
                            counts[TOTAL][docid]['fp_ov'] += 1
                        else:
                            print_debug("    ", docid, ": FALSE POSITIV", ann)

                        if self.subclass_analysis:
                            counts[labelize(ann)][docid]['fp'] += 1
                            if ann in overlap_predicted[labelize(ann)]:
                                counts[labelize(ann)][docid]['fp_ov'] += 1

                for ann in part.annotations:
                    if ann not in part.predicted_annotations:
                        counts[TOTAL][docid]['fn'] += 1

                        if ann in overlap_real[TOTAL]:
                            counts[TOTAL][docid]['fn_ov'] += 1
                        else:
                            print_debug("    ", docid, ": FALSE NEGATIV", ann)

                        if self.subclass_analysis:
                            counts[labelize(ann)][docid]['fn'] += 1
                            if ann in overlap_real[labelize(ann)]:
                                counts[labelize(ann)][docid]['fn_ov'] += 1

        evaluations = Evaluations()

        for label in labels:
            evaluations.add(EvaluationWithStandardError(label, counts[label]))

        return evaluations


class EntityEvaluator(Evaluator):

    TOTAL_LABEL = "TOTAL"

    COMMON_ENTITY_MAP_FUNS = {

        "entity_normalized_fun": (lambda map_entity_normalizations, penalize_unknown_normalizations, add_entity_text:
                                  (lambda e: _entity_normalized_fun(map_entity_normalizations, penalize_unknown_normalizations, add_entity_text, e)))
    }

    def _accept_entities_exact(e1, e2):
        # ASSUME ENTITY TEXT NOT IN STRINGS

        # e.g. e_1|1003,1009|n_7|Q9H4A6
        e1 = e1.split("|")[0:2]
        e2 = e2.split("|")[0:2]
        return e1 == e2


    def _accept_entities_overlapping(e1, e2):
        # e.g. e_1|1003,1009|n_7|Q9H4A6
        e1_class, e1_offsets, *_ = e1.split("|")
        e2_class, e2_offsets, *_ = e2.split("|")

        if e1_class != e2_class:
            return False
        else:
            e1_start_offset, e1_end_offset, *e1_text = e1_offsets.split(',')
            e2_start_offset, e2_end_offset, *e2_text = e2_offsets.split(',')

            return int(e1_start_offset) < int(e2_end_offset) and int(e1_end_offset) > int(e2_start_offset)

    COMMON_ENTITY_ACCEPT_FUNS = {
        'exact': _accept_entities_exact,
        'overlapping': _accept_entities_overlapping
    }


    def __init__(self, entity_map_fun=None, entity_overlap_fun=None, entity_accept_fun=None):

        if entity_map_fun is None:
            self.entity_map_fun = __class__.COMMON_ENTITY_MAP_FUNS["entity_normalized_fun"]({}, penalize_unknown_normalizations="no")
        elif isinstance(entity_map_fun, str):
            assert not entity_map_fun.endswith('_fun'), "You cannot give function names that are complex functions such as 'normalized_fun'"
            self.entity_map_fun = __class__.COMMON_ENTITY_ACCEPT_FUNS["overlapping"]
        else:
            self.entity_map_fun = entity_map_fun

        self.entity_accept_fun = str.__eq__ if entity_accept_fun is None else entity_accept_fun


    @staticmethod
    def _labelize(e):
        if isinstance(e, str):
            match = re.search('Entity\\(id: (\\S+), ', e)
            if match:
                return match.group(1)
            else:
                return e.split("|")[0]
        else:
            return str(e.subclass) if str(e.subclass) not in ['None', 'False'] else str(e.class_id)


    def evaluate(self, dataset):
        """
        :type dataset: nalaf.structures.data.Dataset
        :returns (tp, fp, fn, precision, recall, f_measure): (int, int, int, float, float, float)

        Calculates precision, recall and subsequently F1 measure, defined as:
            * precision: number of correctly predicted items as a percentage of the total number of predicted items
                len(predicted items that are also real)/len(predicted)
                or in other words tp / tp + fp
            * recall: number of correctly predicted items as a percentage of the total number of correct items
                len(real items that are also predicted)/len(real)
                or in other words tp / tp + fn
        """
        TOTAL = EntityEvaluator.TOTAL_LABEL
        labels = [TOTAL]

        # find all possible subclasses or otherwise full classes
        labels += list(set(__class__._labelize(e) for e in dataset.entities()))
        labels += list(set(__class__._labelize(e) for e in dataset.predicted_entities()))

        docids = dataset.documents.keys()
        subcounts = ['tp', 'fp', 'fn']
        counts = {label: {docid: dict.fromkeys(subcounts, 0) for docid in docids} for label in labels}

        for docid, doc in dataset.documents.items():
            for partid, part in doc.parts.items():

                gold_anns = set(filter(None, (self.entity_map_fun(e) for e in part.annotations)))
                pred_anns = set(filter(None, (self.entity_map_fun(e) for e in part.predicted_annotations)))

                for pred in pred_anns:
                    accept_decisions = {self.entity_accept_fun(gold, pred) for gold in gold_anns}
                    assert set.issubset(accept_decisions, {True, False, None}), "did not expect: " + str(accept_decisions)

                    if True in accept_decisions:
                        # Count the true positives while iterating on gold
                        pass

                    elif None in accept_decisions:
                        pass

                    else:
                        # either False or the set is empty, meaning that there are no gold annotations
                        print_debug("    ", docid, ": FALSE POSITIV", pred)
                        counts[TOTAL][docid]['fp'] += 1
                        counts[__class__._labelize(pred)][docid]['fp'] += 1

                for gold in gold_anns:

                    accept_decisions = {self.entity_accept_fun(gold, pred) for pred in pred_anns}

                    if True in accept_decisions:
                        print_verbose("    ", docid, ": true positive", gold)
                        counts[TOTAL][docid]['tp'] += 1
                        counts[__class__._labelize(gold)][docid]['tp'] += 1

                    elif "UNKNOWN:" in gold:  # Pass when unknown normalization
                        pass

                    else:
                        print_debug("    ", docid, ": FALSE NEGATIV", gold)
                        counts[TOTAL][docid]['fn'] += 1
                        counts[__class__._labelize(gold)][docid]['fn'] += 1

        evaluations = Evaluations()

        for label in labels:
            evaluations.add(EvaluationWithStandardError(label, counts[label]))

        return evaluations


def _entity_normalized_fun(map_entity_normalizations, penalize_unknown_normalizations, add_entity_text, e):

    entity_norm_str = _normalized_fun(map_entity_normalizations, penalize_unknown_normalizations, e)
    if entity_norm_str is None:
        return None

    offset_str = [str(e.offset), str(e.end_offset())]
    if add_entity_text:
        offset_str += [e.text]

    offset_str = ','.join(offset_str)

    ret = '|'.join([e.class_id, offset_str, entity_norm_str])
    return ret


def _normalized_fun(map_entity_normalizations, penalize_unknown_normalizations, e):
    n_id = map_entity_normalizations[e.class_id]
    value = e.norms.get(n_id, None)

    if value is None:

        if penalize_unknown_normalizations == "hard":
            # Note: generate random string if norm key is not found to have no dummy clashes out of none keys
            value = "UNKNOWN:" + str(uuid.uuid4())
        elif penalize_unknown_normalizations == "soft":
            value = "UNKNOWN:" + e.text.lower()
        elif penalize_unknown_normalizations == "softest":
            value = "UNKNOWN:" + ""
        elif penalize_unknown_normalizations == "agnostic":
            # returning None (as when "no") would reject the entity altogether, see _entity_normalized_fun
            # returning "" (without UNKNOWN:) simply ignores the case -- Useful when you don't care at all about the normalization (e.g. strict exact / overlapping evaluation)
            value = ""
        elif penalize_unknown_normalizations == "no":
            return None
        else:
            raise AssertionError(("Do not expect: ", penalize_unknown_normalizations))

    return '|'.join([n_id, value])


class DocumentLevelRelationEvaluator(Evaluator):
    """
    Implements document level performance evaluation for relations. It extracts
    and compares unique (gold) relations against the predicted relations.

    The evaluator assumes that all relations are undirected (bidirectional).
    Beware: the assumption is not checked.

    How entities within the relations are compared against each other is decided
    by the user and given in the `entity_map_fun` parameter. The default for this
    is to compare entities by their class id and their text lowercased.
    See other commong comparisons in `__class__.COMMON_ENTITY_MAP_FUNS`.
    You can give this parameter as either a function or as a string name
    that matches one of `COMMON_ENTITY_MAP_FUNS`.

    Furthermore, you can choose how to compare relations to each other.
    First of all, relations are converted to unique strings (together with the
    `entity_map_fun` parameter). Second of all, the default is to compare the
    relation-strings with string's equals function (str.__eq__). However,
    the user can decide with `relation_accept_fun` how to compare the relations' strings.

    More than an equals function, `relation_accept_fun` means:
    * Input: `gold` and `pred` relations in string form (*order matters* --> 1 < 2 but 2 <! 1)
    * Output: decide for `pred` to either, accept (True), reject (False), ignore (None).

    Outputs True and False lead to the expected counts: tp or fp, respectively.
    The special None case is when a prediciton is simply ignored and not counted at all.
    For example, the prediction may be actually correct, yet too vague to be tp, and correct enough to not be a fp.
    As for missing predictions, fn's are counted as usual: `fn` if False or None

    This arbitrary `accept` function is helpful when mere string equals comparison is not enough.
    Example 1: when normalization ids must be compared in a hierarchical manner.
    Example 2: when there are entities / relations without normalization.
    """

    COMMON_ENTITY_MAP_FUNS = {
        'lowercased': (lambda e: '|'.join([str(e.class_id), e.text.lower()])),
        'normalized_fun': (lambda map_entity_normalizations, penalize_unknown_normalizations: (lambda e: _normalized_fun(map_entity_normalizations, penalize_unknown_normalizations, e)))
    }


    def __init__(self, rel_type, entity_map_fun=None, relation_accept_fun=None, evaluate_only_on_edges_plausible_relations=False):
        self.rel_type = rel_type
        if entity_map_fun is None:
            self.entity_map_fun = __class__.COMMON_ENTITY_MAP_FUNS['lowercased']
        elif isinstance(entity_map_fun, str):
            assert not entity_map_fun.endswith('_fun'), "You cannot give function names that are complex functions such as 'normalized_fun'"
            self.entity_map_fun = __class__.COMMON_ENTITY_MAP_FUNS[entity_map_fun]
        else:
            self.entity_map_fun = entity_map_fun

        self.relation_accept_fun = str.__eq__ if relation_accept_fun is None else relation_accept_fun

        self.evaluate_only_on_edges_plausible_relations = evaluate_only_on_edges_plausible_relations


    def evaluate(self, dataset):
        """
        :type dataset: nala.structures.data.Dataset
        :returns Evaluations
        """

        subcounts = ['tp', 'fp', 'fn']
        counts = {docid: dict.fromkeys(subcounts, 0) for docid in dataset.documents.keys()}

        print_verbose()

        for docid, doc in dataset.documents.items():
            if self.evaluate_only_on_edges_plausible_relations:
                # a set would be better, but so far Relation is unshable
                relations_search_space = list(dataset.plausible_relations_from_generated_edges())
            else:
                relations_search_space = None

            gold = doc.map_relations(use_predicted=False, relation_type=self.rel_type, entity_map_fun=self.entity_map_fun, relations_search_space=relations_search_space).keys()
            pred = doc.map_relations(use_predicted=True, relation_type=self.rel_type, entity_map_fun=self.entity_map_fun).keys()

            for r_pred in pred:

                accept_decisions = {self.relation_accept_fun(r_gold, r_pred) for r_gold in gold}
                assert set.issubset(accept_decisions, {True, False, None}), "`relation_accept_fun` cannot return: " + str(accept_decisions)

                if True in accept_decisions:
                    # Count the true positives while iterating on gold
                    pass

                elif None in accept_decisions:
                    # Ignore as documented
                    pass

                else:
                    # either False or the set is empty, meaning that there are no gold annotations
                    print_debug("    ", docid, ": FALSE POSITIV", r_pred)
                    counts[docid]['fp'] += 1

            for r_gold in gold:

                r_preds = [r_pred for r_pred in pred if self.relation_accept_fun(r_gold, r_pred)]

                if len(r_preds) > 0:  # we could also do any(...); we have this in place only for debugging purposes
                    print_verbose("    ", docid, ": true positive", r_gold)
                    counts[docid]['tp'] += 1

                else:
                    print_debug("    ", docid, ": FALSE NEGATIV", r_gold)
                    counts[docid]['fn'] += 1

        print_verbose()

        evaluations = Evaluations()
        evaluations.add(EvaluationWithStandardError(self.rel_type, counts))
        return evaluations
