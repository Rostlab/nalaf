import unittest
from nala.bootstrapping import generate_documents, Iteration
from nose.plugins.attrib import attr


@attr('slow')
class TestBootstrapping(unittest.TestCase):
    def test_generate_documents_number(self):
        # commenting out for now since it takes about 6 mins on Travis CI
        test_dataset = generate_documents(2)
        self.assertEqual(len(test_dataset), 2)
        pass


class TestIteration(unittest.TestCase):
    def test_init(self):
        iteration = Iteration()
        iteration.learning()


if __name__ == '__main__':
    unittest.main()
