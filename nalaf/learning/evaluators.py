import abc
from nalaf.structures.data import Entity
from nalaf import print_verbose, print_debug
from collections import namedtuple
import random
import math

class Evaluation:

    Computation = namedtuple('Computation', ['precision', 'recall', 'f_measure'])

    def __init__(self, label, tp, fp, fn, fp_ov=0, fn_ov=0):
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

            precision = self._safe_div(self.tp + self.fp_ov + self.fn_ov, self.tp + fp + self.fp_ov + self.fn_ov)
            recall = self._safe_div(self.tp + self.fp_ov + self.fn_ov, self.tp + fn + self.fp_ov + self.fn_ov)

        elif strictness == 'half_overlapping':
            fp = self.fp - self.fp_ov
            fn = self.fn - self.fn_ov

            precision = self._safe_div(self.tp + (self.fp_ov + self.fn_ov) / 2, self.tp + fp + self.fp_ov + self.fn_ov)
            recall = self._safe_div(self.tp + (self.fp_ov + self.fn_ov) / 2, self.tp + fn + self.fp_ov + self.fn_ov)

        else:
            raise ValueError('strictness must be "exact" or "overlapping" or "half_overlapping"')

        f_measure = 2 * self._safe_div(precision * recall, precision + recall)

        return Evaluation.Computation(precision, recall, f_measure)

    def __str__(self):
        return self.format()

    def format_header(self, strictnesses=None):
        strictnesses = ['exact', 'overlapping'] if strictnesses is None else strictnesses

        header = ['# class', 'tp', 'fp', 'fn', 'fp_ov', 'fn_ov']
        for _ in strictnesses:
            header += ['match', 'P', 'R', 'F']
        return '\t'.join(header)

    def _format_counts_list(self):
        ret = [self.tp, self.fp, self.fn, self.fp_ov, self.fn_ov]
        return [str(c) for c in ret]

    def format(self, strictnesses=None):
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
            return float('NaN')


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
        self.keys_len = len(dic_counts.keys())
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

        return sum([value[count] for key, value in self.dic_counts.items() if key in keys])

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
                random_keys = random.sample(self.keys, round(self.keys_len * self.p))
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
        return self.format()

    def format_header(self, strictnesses=None):
        strictnesses = ['exact', 'overlapping'] if strictnesses is None else strictnesses

        header = ['# class', 'tp', 'fp', 'fn', 'fp_ov', 'fn_ov']
        for _ in strictnesses:
            header += ['match', 'P', 'P_SE', 'R', 'R_SE', 'F', 'F_SE']
        return '\t'.join(header)

    def format(self, strictnesses=None):
        strictnesses = ['exact', 'overlapping'] if strictnesses is None else strictnesses

        cols = [self.label] + self._mean_eval._format_counts_list()
        for strictness in strictnesses:
            cols += [strictness[0]]  # first character
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

    def format_computation_simple(self, c):
        complist = [c.precision, c.precision_SE, c.recall, c.recall_SE, c.f_measure, c.f_measure_SE]
        return ["{:6.4f}".format(n) for n in complist]


    def format_computation_removing_noise(self, c):
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
            rows += [evaluation.format(strictnesses)]
        return '\n'.join(rows)


    def __iter__(self):
        return self.classes.__iter__()

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
        :returns (precision, recall, f_measure): (float, float, float)
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

        if self.subclass_analysis:
            # find all possible subclasses
            subclasses = set(ann.subclass for ann in dataset.annotations() if ann.subclass is not None)
            subclasses.update(set(ann.subclass for ann in dataset.predicted_annotations() if ann.subclass is not None))
            for x in subclasses:
                labels.append(x)

        docids = dataset.documents.keys()
        subcounts = ['tp', 'fp', 'fn', 'fp_ov', 'fn_ov']
        counts = {label: {docid: dict.fromkeys(subcounts, 0) for docid in docids} for label in labels}

        for docid, doc in dataset.documents.items():
            for partid, part in doc.parts.items():
                tests = ' || '.join(sorted(ann.text for ann in part.annotations))
                preds = ' || '.join(sorted(ann.text for ann in part.predicted_annotations))
                if tests != preds:
                    print_debug("* docid={} part={}".format(docid, partid))
                    print_debug("test: {}".format(tests))
                    print_debug("pred: {}".format(preds))
                    print_debug()

                overlap_real = {label: [] for label in labels}
                overlap_predicted = {label: [] for label in labels}

                Entity.equality_operator = 'overlapping'
                for ann_a in part.annotations:
                    for ann_b in part.predicted_annotations:
                        if ann_a == ann_b:  # equal according according to exclusive overlapping eq (not exact)
                            overlap_real[TOTAL].append(ann_a)
                            overlap_predicted[TOTAL].append(ann_b)

                            if self.subclass_analysis:
                                if ann_a.subclass != ann_b.subclass:
                                    print_debug('overlapping subclasses do not match', ann_a.subclass, ann_b.subclass)
                                    ann_b.subclass = ann_a.subclass

                                overlap_real[ann_a.subclass].append(ann_a)
                                overlap_predicted[ann_b.subclass].append(ann_b)

                Entity.equality_operator = 'exact'
                for ann in part.predicted_annotations:
                    if ann in part.annotations:
                        counts[TOTAL][docid]['tp'] += 1
                        if self.subclass_analysis:
                            counts[ann.subclass][docid]['tp'] += 1
                    else:
                        counts[TOTAL][docid]['fp'] += 1
                        if ann in overlap_predicted[TOTAL]:
                            counts[TOTAL][docid]['fp_ov'] += 1
                        if self.subclass_analysis:
                            counts[ann.subclass][docid]['fp'] += 1
                            if ann in overlap_predicted[ann.subclass]:
                                counts[ann.subclass][docid]['fp_ov'] += 1

                for ann in part.annotations:
                    if ann not in part.predicted_annotations:
                        counts[TOTAL][docid]['fn'] += 1
                        if ann in overlap_real[TOTAL]:
                            counts[TOTAL][docid]['fn_ov'] += 1
                        if self.subclass_analysis:
                            counts[ann.subclass][docid]['fn'] += 1
                            if ann in overlap_real[ann.subclass]:
                                counts[ann.subclass][docid]['fn_ov'] += 1

        evaluations = Evaluations()

        for label in labels:
            evaluations.add(EvaluationWithStandardError(label, counts[label]))

        return evaluations
