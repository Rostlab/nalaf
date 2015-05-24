from features import FeatureGenerator


class SimpleFeatureGenerator(FeatureGenerator):
    def generate(self, dataset):
        template = (-2, -1, 0, 1, 2)

        for sentence in dataset.sentences():
            for index, token in enumerate(sentence):
                for template_index in template:
                    if -1 < index + template_index < len(sentence):
                        token.features['w[%d]' % template_index] = sentence[index+template_index].word