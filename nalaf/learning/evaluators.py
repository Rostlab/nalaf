import abc
from nalaf.structures.data import Entity
from nalaf import print_verbose, print_debug

class Evaluation:

    @staticmethod
    def __safe_division(nominator, denominator):
        try:
            return nominator / denominator
        except ZeroDivisionError:
            return float('NaN')

    def __str__(self):
        return self.format()

    def format_simple(self):
        fs = "{:6.4f}"
        p, r, f = list(map(lambda x: fs.format(x),  [self.precision, self.recall, self.f_measure]))
        l = ["P:"+p, "R:"+r, "F:"+f, self.label, self.strictness]
        return '\t'.join(l)

    def format(self):
        l = [n + ":" + str(v) for n, v in zip(
                ["tp", "fp", "fn", "fpo", "fno"],
                [self.tp, self.fp, self.fn, self.fp_overlap, self.fn_overlap])]

        return '\t'.join(l) + "\t" + self.format_simple()

    def __init__(self, label, strictness, tp, fp, fn, fp_overlap=0, fn_overlap=0):
        self.label = label
        self.strictness = strictness
        self.tp = tp
        self.fp = fp
        self.fn = fn
        self.fp_overlap = fp_overlap
        self.fn_overlap = fn_overlap

        if strictness == 'exact':
            self.precision = self.__safe_division(tp, tp + fp)
            self.recall = self.__safe_division(tp, tp + fn)

        elif strictness == 'overlapping':
            fp = fp - fp_overlap
            fn = fn - fn_overlap

            self.precision = self.__safe_division(tp + fp_overlap + fn_overlap, tp + fp + fp_overlap + fn_overlap)
            self.recall = self.__safe_division(tp + fp_overlap + fn_overlap, tp + fn + fp_overlap + fn_overlap)

        elif strictness == 'half_overlapping':
            fp = fp - fp_overlap
            fn = fn - fn_overlap

            self.precision = self.__safe_division(tp + (fp_overlap + fn_overlap) / 2, tp + fp + fp_overlap + fn_overlap)
            self.recall = self.__safe_division(tp + (fp_overlap + fn_overlap) / 2, tp + fn + fp_overlap + fn_overlap)

        else:
            raise ValueError('strictness must be "exact" or "overlapping" or "half_overlapping"')

        self.f_measure = 2 * self.__safe_division(self.precision * self.recall, self.precision + self.recall)

class Evaluations:
    def __init__(self):
        self.l = []

    def append(self, evaluation):
        self.l.append(evaluation)

    def __iter__(self):
        return self.l.__iter__()

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

    def __init__(self, strictness='exact', subclass_analysis=False):
        self.strictness = strictness
        """
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
        self.subclass_analysis = subclass_analysis
        """
        Whether to report the performance for each subclass separately
        Can be used only with strictness='exact'
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
        tp, fp, fn, fp_overlap, fn_overlap = 0, 0, 0, 0, 0

        if self.subclass_analysis:
            # find all possible subclasses
            subclasses = set(ann.subclass for ann in dataset.annotations())
            subclasses.update(set(ann.subclass for ann in dataset.predicted_annotations()))
            # initialize counts to zero for each subclass
            subclass_counts = {subclass: dict.fromkeys(['tp', 'fp', 'fn', 'fp_overlap', 'fn_overlap'], 0)
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
                            fp_overlap += 1
                        if self.subclass_analysis:
                            subclass_counts[ann.subclass]['fp'] += 1
                            if ann in overlap_subclass_predicted[ann.subclass]:
                                subclass_counts[ann.subclass]['fp_overlap'] += 1

                for ann in part.annotations:
                    if ann not in part.predicted_annotations:
                        fn += 1
                        if ann in overlap_real:
                            fn_overlap += 1
                        if self.subclass_analysis:
                            subclass_counts[ann.subclass]['fn'] += 1
                            if ann in overlap_subclass_real[ann.subclass]:
                                subclass_counts[ann.subclass]['fn_overlap'] += 1

        evaluations = Evaluations()

        if self.subclass_analysis:
            subclass_measures = {}
            for subclass, counts in subclass_counts.items():
                if subclass is None:
                    break
                evaluations.append(
                Evaluation(str(subclass), self.strictness, counts['tp'], counts['fp'], counts['fn'], counts['fp_overlap'], counts['fn_overlap']))

        evaluations.append(Evaluation('TOTAL', self.strictness, tp, fp, fn, fp_overlap, fn_overlap))

        return evaluations
