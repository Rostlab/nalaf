from unittest import TestCase
from nala.bootstrapping.iteration import Iteration

__author__ = 'carsten'


class TestIteration(TestCase):
    def test_learning(self):
        iteration = Iteration('tmpbootstrapping')
        iteration.learning()
