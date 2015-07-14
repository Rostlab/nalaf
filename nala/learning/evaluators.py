import abc


class Evaluator:
    # TODO write nice docstring
    """
    """
    @abc.abstractmethod
    def evaluate(self, dataset):
        """
        :type dataset: nala.structures.data.Dataset
        """
        return


class MentionLevelEvaluator(Evaluator):
    # TODO write nice docstring
    """

    """
    def __init__(self, strictness='exact'):
        self.strictness = strictness

    def evaluate(self, dataset, post_processing_predictions=[]):
        # TODO write nice docstring
        predicted, real, _ = find_offsets(dataset)

        for prediction in post_processing_predictions:
            if not _is_overlapping(prediction, predicted):
                predicted.append(prediction)

        if self.strictness is 'exact':
            precision = sum(1 for item in predicted if item in real)/len(predicted)
            recall = sum(1 for item in real if item in predicted)/len(real)
        elif self.strictness is 'overlapping':
            precision = sum(1 for item in predicted if _is_overlapping(item, real))/len(predicted)
            recall = sum(1 for item in real if _is_overlapping(item, predicted))/len(real)
        elif self.strictness is 'half_overlapping':
            precision = sum(1 if item in real else 0.5 if _is_overlapping(item, real) else 0
                            for item in predicted)/len(predicted)
            recall = sum(1 if item in predicted else 0.5 if _is_overlapping(item, predicted) else 0
                         for item in real)/len(real)

        f_measure = 2 * (precision * recall) / (precision + recall)

        print('p:{:.4f} r:{:.4f} f:{:.4f} strictness:{} '.format(precision, recall, f_measure, self.strictness))
        return precision, recall, f_measure


def find_offsets(dataset):
    # TODO write nice docstring
    """
    :type dataset: nala.structures.data.Dataset
    """
    predicted_offsets = []
    real_offsets = []
    predicted_items = []
    for part_id, part in dataset.partids_with_parts():
        so_far = 0
        for sentence in part.sentences:
            index = 0
            while index < len(sentence):
                token = sentence[index]
                so_far = part.text.find(token.word, so_far)

                if token.predicted_labels[0].value is not 'O':
                    start = so_far
                    while index + 1 < len(sentence) and sentence[index + 1].predicted_labels[0].value is not 'O':
                        token = sentence[index + 1]
                        so_far = part.text.find(token.word, so_far)
                        index += 1
                    end = so_far + len(token.word)
                    predicted_offsets.append((start, end, part_id))
                    predicted_items.append(part.text[start:end])
                index += 1
        for ann in part.annotations:
            real_offsets.append((ann.offset, ann.offset + len(ann.text), part_id))

    return predicted_offsets, real_offsets, predicted_items


def _is_overlapping(offset_a, offset_list):
    # TODO write nice docstring
    """
    :param offset_a:
    :param offset_list:
    :return:
    """
    for offset_b in offset_list:
        # (part_idA == part_idB) and (StartA <= EndB)  and  (EndA >= StartB)
        if offset_a[2] == offset_b[2] and offset_a[0] <= offset_b[1] and offset_a[1] >= offset_b[0]:
            return True
    return False