from nalaf.features import FeatureGenerator


class ConjunctionFeatureGenerator(FeatureGenerator):
    """

    """
    def __init__(self, conjunctions):
        self.conjunctions = conjunctions

    def generate(self, dataset):
        """
        :type dataset: nalaf.structures.data.Dataset
        """
        for token in dataset.tokens():
            for conjunction in self.conjunctions:
                token.features['|'.join(conjunction)] = '|'.join(str(token.features.get(item)) for item in conjunction)