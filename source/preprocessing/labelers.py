import abc

class Labeler():
    """
    Abstract class for generating labels for each token in the dataset.
    Subclasses that inherit this class should:
    * Be named [Name]Labeler
    * Implement the abstract method label
    * Append new items to the list field "original_labels" of each Token in the dataset
    """
    @abc.abstractmethod
    def label(self, dataset):
        """
        :type dataset: structures.data.Dataset
        """
        return


class SimpleLabeler(Labeler):
    """
    Implements a simple labeler using the annotations of the dataset
    using the BIO (beginning, inside, outside) format. Creates labels
    based on the class_id value in the Annotation object. That is:
    * B-[class_id]
    * I-[class_id]
    * O

    Requires the list field "annotations" to be previously set.
    Implements the abstract class Labeler.
    """
    def label(self, dataset):
        """
        :type dataset: structures.data.Dataset
        """
        for part in dataset.parts():
            so_far = 0
            for sentence in part.sentences:
                for token in sentence:
                    so_far = part.text.find(token.word, so_far)
                    token.original_labels = [Label('O')]

                    for ann in part.annotations:
                        start = ann.offset
                        end = ann.offset + len(ann.text)
                        if start == so_far:
                            token.original_labels[0].value = 'B-%s' % ann.class_id
                            break
                        elif start < so_far < end:
                            token.original_labels[0].value = 'I-%s' % ann.class_id
                            break
