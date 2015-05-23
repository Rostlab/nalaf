import abc


class FeatureGenerator():
    @abc.abstractmethod
    def generate(self, dataset):
        return