import abc
from nalaf.structures.data import Entity
from nalaf import print_verbose, print_debug
from collections import namedtuple

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
            precision = self.__safe_div(self.tp, self.tp + self.fp)
            recall = self.__safe_div(self.tp, self.tp + self.fn)

        elif strictness == 'overlapping':
            fp = self.fp - self.fp_ov
            fn = self.fn - self.fn_ov

            precision = self.__safe_div(self.tp + self.fp_ov + self.fn_ov, self.tp + fp + self.fp_ov + self.fn_ov)
            recall = self.__safe_div(self.tp + self.fp_ov + self.fn_ov, self.tp + fn + self.fp_ov + self.fn_ov)

        elif strictness == 'half_overlapping':
            fp = self.fp - self.fp_ov
            fn = self.fn - self.fn_ov

            precision = self.__safe_div(self.tp + (self.fp_ov + self.fn_ov) / 2, self.tp + fp + self.fp_ov + self.fn_ov)
            recall = self.__safe_div(self.tp + (self.fp_ov + self.fn_ov) / 2, self.tp + fn + self.fp_ov + self.fn_ov)

        else:
            raise ValueError('strictness must be "exact" or "overlapping" or "half_overlapping"')

        f_measure = 2 * self.__safe_div(precision * recall, precision + recall)

        return Evaluation.Computation(precision, recall, f_measure)

    def __str__(self):
        return self.format()

    def __computation_to_list(self, d):
        return [d.precision, d.recall, d.f_measure]

    def format_header(self, strictnesses=['exact', 'overlapping']):
        header = ['label', 'tp', 'fp', 'fn', 'fp_ov', 'fn_ov']
        for _ in strictnesses:
            header += ['match', 'P', 'R', 'F']
        return '\t'.join(header)

    def format(self, strictnesses=['exact', 'overlapping']):
        l = [self.label]
        counts = [self.tp, self.fp, self.fn, self.fp_ov, self.fn_ov]
        counts = [str(c) for c in counts]
        l += counts
        for strictness in strictnesses:
            l += [strictness[0]]
            l += self.format_computation(self.__computation_to_list(self.compute(strictness)))
        return '\t'.join(l)

    def format_computation(self, computationlist):
        return ["{:6.4f}".format(n) for n in computationlist]

    @staticmethod
    def __safe_div(nominator, denominator):
        try:
            return nominator / denominator
        except ZeroDivisionError:
            return float('NaN')


class Evaluations:

    def __init__(self):
        self.classes = {}

    def __call__(self, clazz):
        return self.classes[str(clazz)]

    def add(self, evaluation):
        self.classes[str(evaluation.label)] = evaluation

    def __str__(self):
        return self.format()

    def format(self, strictnesses=['exact', 'overlapping']):
        assert(len(self.classes) >= 1)
        l = [next(iter(self.classes.values())).format_header(strictnesses)]
        for clazz in sorted(self.classes.keys()):
            evaluation = self.classes[clazz]
            l += [evaluation.format(strictnesses)]
        return '\n'.join(l)


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
        tp, fp, fn, fp_ov, fn_ov = 0, 0, 0, 0, 0

        if self.subclass_analysis:
            # find all possible subclasses
            subclasses = set(ann.subclass for ann in dataset.annotations())
            subclasses.update(set(ann.subclass for ann in dataset.predicted_annotations()))
            # initialize counts to zero for each subclass
            subclass_counts = {subclass: dict.fromkeys(['tp', 'fp', 'fn', 'fp_ov', 'fn_ov'], 0)
                               for subclass in subclasses}

        for doc in dataset:
            for part in doc:
                print_debug(' || '.join(ann.text for ann in part.annotations))
                print_debug(' || '.join(ann.text for ann in part.predicted_annotations))
                print_debug()

                overlap_real = []
                overlap_predicted = []

                if self.subclass_analysis:
                    overlap_subclass_real = {subclass: [] for subclass in subclasses}
                    overlap_subclass_predicted = {subclass: [] for subclass in subclasses}

                Entity.equality_operator = 'overlapping'
                for ann_a in part.annotations:
                    for ann_b in part.predicted_annotations:
                        if ann_a == ann_b:
                            overlap_real.append(ann_a)
                            overlap_predicted.append(ann_b)

                            if self.subclass_analysis:
                                if ann_a.subclass != ann_b.subclass:
                                    print_debug('overlapping subclasses do not match', ann_a.subclass, ann_b.subclass)
                                    ann_b.subclass = ann_a.subclass

                                overlap_subclass_real[ann_a.subclass].append(ann_a)
                                overlap_subclass_predicted[ann_b.subclass].append(ann_b)

                Entity.equality_operator = 'exact'
                for ann in part.predicted_annotations:
                    if ann in part.annotations:
                        tp += 1
                        if self.subclass_analysis:
                            subclass_counts[ann.subclass]['tp'] += 1
                    else:
                        fp += 1
                        if ann in overlap_predicted:
                            fp_ov += 1
                        if self.subclass_analysis:
                            subclass_counts[ann.subclass]['fp'] += 1
                            if ann in overlap_subclass_predicted[ann.subclass]:
                                subclass_counts[ann.subclass]['fp_ov'] += 1

                for ann in part.annotations:
                    if ann not in part.predicted_annotations:
                        fn += 1
                        if ann in overlap_real:
                            fn_ov += 1
                        if self.subclass_analysis:
                            subclass_counts[ann.subclass]['fn'] += 1
                            if ann in overlap_subclass_real[ann.subclass]:
                                subclass_counts[ann.subclass]['fn_ov'] += 1

        evaluations = Evaluations()

        if self.subclass_analysis:
            for subclass, counts in subclass_counts.items():
                if subclass is None:
                    break
                evaluations.add(Evaluation(str(subclass), counts['tp'], counts['fp'], counts['fn'], counts['fp_ov'], counts['fn_ov']))

        evaluations.add(Evaluation('TOTAL', tp, fp, fn, fp_ov, fn_ov))

        return evaluations
