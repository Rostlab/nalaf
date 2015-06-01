from nala.features import FeatureGenerator


class SimpleFeatureGenerator(FeatureGenerator):
    """
    Generates simple CRF features based on the value of the token itself.
    For each token t, generates at most 5 features, corresponding to:
    * w[0] = the value of the words itself
    * w[1], w[2] = the values of the next two words in the sequence
    * w[-1], w[-2] = the values of the previous two words in the sequence

    Implements the abstract class FeatureGenerator.

    TODO: Instead of having a hard-coded template, allow the template to
    be optionally passed as a parameter.
    """

    def generate(self, dataset):
        """
        :type dataset: structures.data.Dataset
        """
        template = (-2, -1, 0, 1, 2)

        for sentence in dataset.sentences():
            for index, token in enumerate(sentence):
                for template_index in template:
                    if -1 < index + template_index < len(sentence):
                        token.features['w[%d]' % template_index] = sentence[index + template_index].word
