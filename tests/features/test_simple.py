import unittest
from nalaf.structures.data import Dataset, Document, Part, Token
from nalaf.features.simple import SimpleFeatureGenerator, SentenceMarkerFeatureGenerator


class TestSimpleFeatureGenerator(unittest.TestCase):
    def setUp(self):
        part = Part('Word1 word2 word3. Word4 word5 word6.')
        part.sentences = [[Token('Word1', 0), Token('word2', 6), Token('word3', 12)],
                          [Token('Word4', 19), Token('word5', 25), Token('word6', 31)]]

        self.dataset = Dataset()
        self.dataset.documents['doc_1'] = Document()
        self.dataset.documents['doc_1'].parts['part_1'] = part

        self.simple_generator = SimpleFeatureGenerator()
        self.sentence_generator = SentenceMarkerFeatureGenerator()

    def test_simple_generate(self):
        self.simple_generator.generate(self.dataset)
        features = [token.features for token in self.dataset.tokens()]
        expected = iter([{'word[0]': 'Word1'}, {'word[0]': 'word2'}, {'word[0]': 'word3'},
                         {'word[0]': 'Word4'}, {'word[0]': 'word5'}, {'word[0]': 'word6'}])
        for feature in features:
            self.assertEqual(feature, next(expected))

    def test_sentence_generate(self):
        self.sentence_generator.generate(self.dataset)
        features = [token.features for token in self.dataset.tokens()]
        expected = iter([{'BOS[0]': 1}, {}, {'EOS[0]': 1}, {'BOS[0]': 1}, {}, {'EOS[0]': 1}])

        for feature in features:
            self.assertEqual(feature, next(expected))

if __name__ == '__main__':
    unittest.main()
