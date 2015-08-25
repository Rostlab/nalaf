import abc
from nala.structures.data import Annotation
from nala import print_verbose, print_debug


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
        :type dataset: nala.structures.data.Dataset
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
        :type dataset: nala.structures.data.Dataset
        :returns (tp, fp, fn, tp_overlapping, precision, recall, f_measure): (int, int, int, int, float, float, float)

        Calculates precision, recall and subsequently F1 measure, defined as:
            * precision: number of correctly predicted items as a percentage of the total number of predicted items
                len(predicted items that are also real)/len(predicted)
                or in other words tp / tp + fp
            * recall: number of correctly predicted items as a percentage of the total number of correct items
                len(real items that are also predicted)/len(real)
                or in other words tp / tp + fn
            * possibly considers overlapping matches as well

        Also prints the value of the calculated precision, recall, F1 measure
        as well as the value of the parameter 'strictness'.
        """
        tp, fp, fn, tp_overlapping = 0, 0, 0, 0

        if self.subclass_analysis:
            # find all possible subclasses
            subclasses = set(ann.subclass for ann in dataset.annotations())
            subclasses.update(set(ann.subclass for ann in dataset.predicted_annotations()))
            # initialize counts to zero for each subclass
            subclass_counts = {subclass: dict.fromkeys(['tp', 'fp', 'fn', 'tp_overlapping'], 0)
                               for subclass in subclasses}

        for doc in dataset:
            for part in doc:
                print_debug(' || '.join(ann.text for ann in part.annotations))
                print_debug(' || '.join(ann.text for ann in part.predicted_annotations))
                print_debug()

                Annotation.equality_operator = 'exact'
                for ann in part.predicted_annotations:
                    if ann in part.annotations:
                        tp += 1
                        if self.subclass_analysis:
                            subclass_counts[ann.subclass]['tp'] += 1
                    else:
                        fp += 1
                        if self.subclass_analysis:
                            subclass_counts[ann.subclass]['fp'] += 1

                for ann in part.annotations:
                    if ann not in part.predicted_annotations:
                        fn += 1
                        if self.subclass_analysis:
                            subclass_counts[ann.subclass]['fn'] += 1

                Annotation.equality_operator = 'overlapping'
                for ann_a in part.annotations:
                    for ann_b in part.predicted_annotations:
                        if ann_a == ann_b:
                            tp_overlapping += 1
                            if self.subclass_analysis and ann_a.subclass == ann_b.subclass:
                                subclass_counts[ann_a.subclass]['tp_overlapping'] += 1

        if self.subclass_analysis:
            for subclass, counts in subclass_counts.items():
                print('SUBCLASS {:4}'.format(subclass), end='\t')
                self.__calc_measures(counts['tp'], counts['fp'], counts['fn'], counts['tp_overlapping'])
            print('TOTAL'.ljust(14), end='\t')

        return self.__calc_measures(tp, fp, fn, tp_overlapping)

    @staticmethod
    def __safe_division(nominator, denominator):
        try:
            return nominator / denominator
        except ZeroDivisionError:
            return float('NaN')

    def __calc_measures(self, tp, fp, fn, tp_overlapping):
        if self.strictness == 'exact':
            precision = self.__safe_division(tp, tp + fp)
            recall = self.__safe_division(tp, tp + fn)
        elif self.strictness == 'overlapping':
            fp = fp - tp_overlapping
            fn = fn - tp_overlapping

            precision = self.__safe_division(tp + tp_overlapping, tp + fp + tp_overlapping)
            recall = self.__safe_division(tp + tp_overlapping, tp + fn + tp_overlapping)
        elif self.strictness == 'half_overlapping':
            fp = fp - tp_overlapping
            fn = fn - tp_overlapping

            precision = self.__safe_division(tp + tp_overlapping / 2, tp + fp + tp_overlapping)
            recall = self.__safe_division(tp + tp_overlapping / 2, tp + fn + tp_overlapping)
        else:
            raise ValueError('strictness must be "exact" or "overlapping" or "half_overlapping"')

        f_measure = 2 * self.__safe_division(precision * recall, precision + recall)

        print_verbose('tp:{:4} fp:{:4} fn:{:4} tp_overlapping:{:4} '
                      .format(tp, fp, fn, tp_overlapping, precision, recall, f_measure, self.strictness))

        print('p:{:.4f} r:{:.4f} f:{:.4f} strictness:{} '
              .format(precision, recall, f_measure, self.strictness))
        return tp, fp, fn, tp_overlapping, precision, recall, f_measure
