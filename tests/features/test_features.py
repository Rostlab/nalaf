import unittest
from nalaf.features import eval_binary_feature
from nalaf.structures.data import FeatureDictionary
import re


class TestBinaryFeatureWrapper(unittest.TestCase):
    def test_regex_evaluator(self):
        regex = re.compile('^[A-Z]+$')

        feature_dict = FeatureDictionary()
        eval_binary_feature(feature_dict, 'name', regex.search, 'ABC')

        self.assertEqual(feature_dict.get('name[0]'), 1)
        self.assertEqual(len(feature_dict), 1)

        feature_dict = FeatureDictionary()
        eval_binary_feature(feature_dict, 'name', regex.search, 'abc')

        self.assertEqual(feature_dict.get('name[0]'), None)
        self.assertEqual(len(feature_dict), 0)

    def test_lambda_evaluator(self):
        feature_dict = FeatureDictionary()
        eval_binary_feature(feature_dict, 'name', lambda x: x == 'ABC', 'ABC')
        self.assertEqual(feature_dict.get('name[0]'), 1)
        self.assertEqual(len(feature_dict), 1)

        feature_dict = FeatureDictionary()
        eval_binary_feature(feature_dict, 'name', lambda x: x == 'ABC', 'abc')
        self.assertEqual(feature_dict.get('name[0]'), None)
        self.assertEqual(len(feature_dict), 0)

        feature_dict = FeatureDictionary()
        eval_binary_feature(feature_dict, 'name', lambda x, y: x == y, 'xx', 'xx')
        self.assertEqual(feature_dict.get('name[0]'), 1)
        self.assertEqual(len(feature_dict), 1)

        feature_dict = FeatureDictionary()
        eval_binary_feature(feature_dict, 'name', lambda x, y: x == y, 'xx', 'yy')
        self.assertEqual(feature_dict.get('name[0]'), None)
        self.assertEqual(len(feature_dict), 0)


if __name__ == '__main__':
    unittest.main()
