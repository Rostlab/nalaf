import unittest
from nalaf.structures.data import Dataset, Document, Part
from nalaf.preprocessing.spliters import NLTKSplitter


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

        sentences_ = []
        for document in self.dataset.documents.values():
            for part in document.parts.values():
                sentences_ += part.sentences_

        expected = ['This is one sentence.', 'This is another one.', 'This is the third one; here continues.']

        self.assertEqual(sentences_, expected)


if __name__ == '__main__':
    unittest.main()
