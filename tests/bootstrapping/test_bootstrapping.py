import unittest
from nala.bootstrapping import generate_documents


class TestBootstrapping(unittest.TestCase):
    def test_generate_documents_number(self):
        test_dataset = generate_documents(1)
        self.assertEqual(len(test_dataset), 1)


if __name__ == '__main__':
    unittest.main()
