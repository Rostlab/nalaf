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

# TODO Rename implementation class to include Impl in the name
class MentionLevelEvaluator(Evaluator):
    """
    Implements mention level performance evaluation. That means it compares if the predicted text spans match
    the original annotated text spans.

    Whether a text spans matches and how we count that match is determined
    by the value of the parameter 'strictness'.
    """

    def __init__(self, strictness='exact'):
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
        :returns (precision, recall, f_measure): (float, float, float)

        Calculates precision, recall and subsequently F1 measure, defined as:
            * precision: number of correctly predicted items as a percentage of the total number of predicted items
                len(predicted items that are also real)/len(predicted)
            * recall: number of correctly predicted items as a percentage of the total number of correct items
                len(real items that are also predicted)/len(real)

        Also prints the value of the calculated precision, recall, F1 measure
        as well as the value of the parameter 'strictness'.
        """
        tp, fp, fn, tp_overlapping = 0, 0, 0, 0
        for doc in dataset:
            for part in doc:
                Annotation.strictness = 'exact'
                tp += sum(1 for ann in part.predicted_annotations if ann in part.annotations)
                fp += sum(1 for ann in part.predicted_annotations if ann not in part.annotations)
                fn += sum(1 for ann in part.annotations if ann not in part.predicted_annotations)
                Annotation.strictness = 'overlapping'
                for ann_a in part.annotations:
                    for ann_b in part.predicted_annotations:
                        if ann_a == ann_b:
                            tp_overlapping += 1

        if self.strictness == 'exact':
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
        elif self.strictness == 'overlapping':
            precision = (tp + tp_overlapping) / (tp + fp + tp_overlapping/2)
            recall = (tp + tp_overlapping) / (tp + fn + tp_overlapping/2)
        elif self.strictness == 'half_overlapping':
            precision = (tp + tp_overlapping/2) / (tp + fp + tp_overlapping/2)
            recall = (tp + tp_overlapping/2) / (tp + fn + tp_overlapping/2)

        f_measure = 2 * (precision * recall) / (precision + recall)
        print('p:{:.4f} r:{:.4f} f:{:.4f} strictness:{} '.format(precision, recall, f_measure, self.strictness))
        return precision, recall, f_measure


def find_offsets(dataset):
    """
    :type dataset: nala.structures.data.Dataset

    Forms tuples of 5 elements representing the offsets of the real and predicted annotations
    used in the MentionLevelEvaluator.

    The tuple is (start, end, part_id, class_id, doc_id).
    """
    real_offsets = []
    predicted_offsets = []


    for doc_id, doc in dataset.documents.items():
        for part_id, part in doc.parts.items():
            for ann in part.annotations:
                real_offsets.append((ann.offset, ann.offset + len(ann.text), part_id, ann.class_id, doc_id))
            for ann in part.predicted_annotations:
                predicted_offsets.append((ann.offset, ann.offset + len(ann.text), part_id, ann.class_id, doc_id))

    return real_offsets, predicted_offsets


def is_overlapping(offset_a, offset_list):
    """
    :param offset_a:
    :param offset_list:
    :return:

    Determines if there is an overlapping offset as defined by find_offsets().

    Returns True if offset_a is overlapping with ANY of the offsets in offset_list,
    Returns False otherwise, where overlapping is defined as:
        * class_id and part_id and doc_id are equal
        * and the range is overlapping, where overlapping range is defined as:

        Let condition_1 mean that range_a is completely after range_b
            True if StartA > EndB
        Let condition_2 mean that range_a is completely before range_b
            True if EndA < StartB
        Then overlap exists if neither condition 1 nor 2 is true
            not (condition_1 or condition_2)
            which translates to (StartA <= EndB)  and  (EndA >= StartB)

    """
    # TODO Clean up this; separate function
    for offset_b in offset_list:
        # class_id and part_id and doc_id are equal and (StartA <= EndB) and (EndA >= StartB)
        if offset_a[2:5] == offset_b[2:5] and offset_a[0] <= offset_b[1] and offset_a[1] >= offset_b[0]:
            return True
    return False
