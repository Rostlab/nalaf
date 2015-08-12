import abc
from nala.structures.data import Annotation


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

    # TODO put verbose (and or debug) flags in Config object
    def __init__(self, strictness='exact', verbose=False):
        self.verbose = verbose
        """
        Enables printing of extra information used for debugging such as
        the real and prediction annotations for each part in a separate line
        """
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
        for doc in dataset:
            for part in doc:
                if self.verbose:
                    print(' || '.join(ann.text for ann in part.annotations))
                    print(' || '.join(ann.text for ann in part.predicted_annotations))
                    print()

                Annotation.equality_operator = 'exact'
                tp += sum(1 for ann in part.predicted_annotations if ann in part.annotations)
                fp += sum(1 for ann in part.predicted_annotations if ann not in part.annotations)
                fn += sum(1 for ann in part.annotations if ann not in part.predicted_annotations)
                Annotation.equality_operator = 'overlapping'
                for ann_a in part.annotations:
                    for ann_b in part.predicted_annotations:
                        if ann_a == ann_b:
                            tp_overlapping += 1

        if self.strictness == 'exact':
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
        elif self.strictness == 'overlapping':
            fp = fp - tp_overlapping
            fn = fn - tp_overlapping

            precision = (tp + tp_overlapping) / (tp + fp + tp_overlapping)
            recall = (tp + tp_overlapping) / (tp + fn + tp_overlapping)
        elif self.strictness == 'half_overlapping':
            fp = fp - tp_overlapping
            fn = fn - tp_overlapping

            precision = (tp + tp_overlapping / 2) / (tp + fp + tp_overlapping)
            recall = (tp + tp_overlapping / 2) / (tp + fn + tp_overlapping)
        else:
            raise ValueError('strictness must be "exact" or "overlapping" or "half_overlapping"')

        f_measure = 2 * (precision * recall) / (precision + recall)

        print('p:{:.4f} r:{:.4f} f:{:.4f} strictness:{} '.format(precision, recall, f_measure, self.strictness))
        return tp, fp, fn, tp_overlapping, precision, recall, f_measure
