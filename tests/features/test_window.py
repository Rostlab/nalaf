import unittest
from nalaf.structures.data import Dataset, Document, Part, Token
from nalaf.features.window import WindowFeatureGenerator


class TestWindowFeatureGenerator(unittest.TestCase):
    def setUp(self):
        part = Part('Make making made. Try tried tries.')
        part.sentences = [[Token('Make', 0), Token('making', 5), Token('made', 12)],
                          [Token('Try', 18), Token('tried', 22), Token('tries', 28)]]
        self.dataset = Dataset()
        self.dataset.documents['doc_1'] = Document()
        self.dataset.documents['doc_1'].parts['part_1'] = part

        for token in self.dataset.tokens():
            token.features['a'] = 'a'
            token.features['b'] = 'b'

    def test_default_window(self):
        WindowFeatureGenerator().generate(self.dataset)
        sentences = self.dataset.documents['doc_1'].parts['part_1'].sentences

        self.assertEqual(sentences[0][0].features, {'a[0]': 'a', 'a[1]': 'a', 'a[2]': 'a',
                                                    'b[0]': 'b', 'b[1]': 'b', 'b[2]': 'b'})
        self.assertEqual(sentences[0][1].features, {'a[-1]': 'a', 'a[0]': 'a', 'a[1]': 'a',
                                                    'b[-1]': 'b', 'b[0]': 'b', 'b[1]': 'b'})
        self.assertEqual(sentences[0][2].features, {'a[-2]': 'a', 'a[-1]': 'a', 'a[0]': 'a',
                                                    'b[-2]': 'b', 'b[-1]': 'b', 'b[0]': 'b'})
        self.assertEqual(sentences[1][0].features, {'a[0]': 'a', 'a[1]': 'a', 'a[2]': 'a',
                                                    'b[0]': 'b', 'b[1]': 'b', 'b[2]': 'b'})
        self.assertEqual(sentences[1][1].features, {'a[-1]': 'a', 'a[0]': 'a', 'a[1]': 'a',
                                                    'b[-1]': 'b', 'b[0]': 'b', 'b[1]': 'b'})
        self.assertEqual(sentences[1][2].features, {'a[-2]': 'a', 'a[-1]': 'a', 'a[0]': 'a',
                                                    'b[-2]': 'b', 'b[-1]': 'b', 'b[0]': 'b'})

    def test_custom_window(self):
        WindowFeatureGenerator(template=(-2, 1)).generate(self.dataset)
        sentences = self.dataset.documents['doc_1'].parts['part_1'].sentences
        self.assertEqual(sentences[0][0].features, {'a[0]': 'a', 'a[1]': 'a', 'b[0]': 'b', 'b[1]': 'b'})
        self.assertEqual(sentences[0][1].features, {'a[0]': 'a', 'a[1]': 'a', 'b[0]': 'b', 'b[1]': 'b'})
        self.assertEqual(sentences[0][2].features, {'a[-2]': 'a', 'a[0]': 'a', 'b[-2]': 'b', 'b[0]': 'b'})
        self.assertEqual(sentences[1][0].features, {'a[0]': 'a', 'a[1]': 'a', 'b[0]': 'b', 'b[1]': 'b'})
        self.assertEqual(sentences[1][1].features, {'a[0]': 'a', 'a[1]': 'a', 'b[0]': 'b', 'b[1]': 'b'})
        self.assertEqual(sentences[1][2].features, {'a[-2]': 'a', 'a[0]': 'a', 'b[-2]': 'b', 'b[0]': 'b'})

    def test_include_list(self):
        WindowFeatureGenerator(include_list=['a[0]']).generate(self.dataset)
        sentences = self.dataset.documents['doc_1'].parts['part_1'].sentences

        self.assertEqual(sentences[0][0].features, {'a[0]': 'a', 'a[1]': 'a', 'a[2]': 'a', 'b[0]': 'b'})
        self.assertEqual(sentences[0][1].features, {'a[-1]': 'a', 'a[0]': 'a', 'a[1]': 'a', 'b[0]': 'b'})
        self.assertEqual(sentences[0][2].features, {'a[-2]': 'a', 'a[-1]': 'a', 'a[0]': 'a', 'b[0]': 'b'})

        self.assertEqual(sentences[1][0].features, {'a[0]': 'a', 'a[1]': 'a', 'a[2]': 'a', 'b[0]': 'b'})
        self.assertEqual(sentences[1][1].features, {'a[-1]': 'a', 'a[0]': 'a', 'a[1]': 'a', 'b[0]': 'b'})
        self.assertEqual(sentences[1][2].features, {'a[-2]': 'a', 'a[-1]': 'a', 'a[0]': 'a', 'b[0]': 'b'})

if __name__ == '__main__':
    unittest.main()
