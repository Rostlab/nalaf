import abc


class Labeler():
    @abc.abstractmethod
    def label(self, dataset):
        return


class SimpleLabeler(Labeler):
    def label(self, dataset):
        for part in dataset.parts():
            so_far = 0
            for sentence in part.sentences:
                for token in sentence:
                    so_far = part.text.find(token.word, so_far)
                    token.original_labels = ['O']

                    for ann in part.annotations:
                        start = ann.offset
                        end = ann.offset + len(ann.text)
                        if start == so_far:
                            token.original_labels[0] = 'B-%s' % ann.class_id
                            break
                        elif start < so_far < end:
                            token.original_labels[0] = 'I-%s' % ann.class_id
                            break
