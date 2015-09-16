import unittest
from nala.structures.data import Dataset, Document, Part
from nala.preprocessing.spliters import NLTKSplitter


class TestNLTKSplitter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = Dataset()
        doc = Document()
        part = Part('This is one sentence. This is another one.\n This is the third one; here continues.')
        cls.dataset.documents['doc_1'] = doc
        doc.parts['part_1'] = part

    def test_split(self):
        NLTKSplitter().split(self.dataset)
        sentences = list(self.dataset.sentences())
        expected = ['This is one sentence.', 'This is another one.', 'This is the third one; here continues.']
        self.assertEqual(sentences, expected)


if __name__ == '__main__':
    unittest.main()
