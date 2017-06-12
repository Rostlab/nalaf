import unittest
from nalaf.structures.data import Dataset, Document, Part, Token, Label, Entity
from nalaf.preprocessing.spliters import NLTKSplitter
from nalaf.preprocessing.tokenizers import NLTK_TOKENIZER
from nalaf import print_verbose, print_debug
from nalaf.utils.readers import StringReader
from nalaf.preprocessing.parsers import Parser, SpacyParser
from nalaf.features import get_spacy_nlp_english
# from nose.plugins.attrib import attr


STUB_ENTITY_CLASS_ID = 'e_x'


class TestDataset(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dataset = Dataset()
        cls.doc = Document()
        cls.dataset.documents['testid'] = cls.doc

        # TEXT = "123 45678"
        # POS  = "012345678"
        # ANN1 = " X       "
        # ANN2 = "     XXX "
        # PAR1 = "XXX      "
        # PAR1 = "    XXXXX"

        part1 = Part('123')
        part2 = Part('45678')
        ann1 = Entity(class_id=STUB_ENTITY_CLASS_ID, offset=1, text='2', confidence=0)
        ann2 = Entity(class_id=STUB_ENTITY_CLASS_ID, offset=1, text='567', confidence=1)
        ann1.subclass = 0
        ann2.subclass = 2
        part1.annotations.append(ann1)
        part2.annotations.append(ann2)
        cls.doc.parts['s1h1'] = part1
        cls.doc.parts['s2p1'] = part2

        doc2 = Document()
        doc3 = Document().parts['someid'] = Part('marmor stein und eisen')
        cls.dataset2 = Dataset()
        cls.dataset2.documents['newid'] = doc3
        cls.dataset2.documents['testid'] = doc2

    def test_repr_full(self):
        print(str(self.dataset))

    def test_overlaps_with_mention(self):
        # True states
        self.assertTrue(self.doc.overlaps_with_mention(5, 5))
        self.assertTrue(self.doc.overlaps_with_mention(6, 8))
        self.assertTrue(self.doc.overlaps_with_mention(4, 5))
        self.assertTrue(self.doc.overlaps_with_mention(1, 7))

        # False states
        self.assertFalse(self.doc.overlaps_with_mention(3, 4))
        self.assertFalse(self.doc.overlaps_with_mention(0, 0))
        self.assertFalse(self.doc.overlaps_with_mention(2, 4))

    def test_get_title(self):
        self.assertEquals(self.doc.get_title(), '123')

    def test_get_text(self):
        self.assertEquals(self.doc.get_text(), '123 45678')

    def test_get_body(self):
        self.assertEquals(self.doc.get_body(), '45678')

    def test_get_size(self):
        self.assertEquals(self.doc.get_size(), 9)

    def test_extend_dataset(self):
        print(self.dataset)
        print("\n\n\n")
        self.dataset.extend_dataset(self.dataset2)
        print(self.dataset)

    def test_delete_subclass_annotations(self):
        self.dataset.delete_subclass_annotations(0)
        self.assertEqual(len(list(self.dataset.annotations())), 1)


class TestDocument(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        text1 = "Flowers in the Rain. Are absolutely marvellous. Though i would say this text is stupid. Cheers!"

        part1 = Part(text1)
        doc = Document()
        doc.parts['firstpart'] = part1
        dataset = Dataset()
        dataset.documents['firstdocument'] = doc

        NLTKSplitter().split(dataset)
        # TmVarTokenizer().tokenize(dataset)
        cls.data = dataset
        cls.testpart = dataset.documents['firstdocument'].parts['firstpart']

    def test_get_size(self):
        self.assertEqual(self.testpart.get_size(), 95)

    def test_get_sentence_string_array(self):
        self.assertEqual(self.testpart.get_sentence_string_array(),
                         ["Flowers in the Rain.", "Are absolutely marvellous.",
                          "Though i would say this text is stupid.", "Cheers!"])


class TestToken(unittest.TestCase):
    pass


class TestFeatureDictionary(unittest.TestCase):
    pass


class TestEntity(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dataset = Dataset()
        cls.doc = Document()
        cls.dataset.documents['testid'] = cls.doc

        # TEXT = "123 45678"
        # POS  = "012345678"
        # ANN1 = " X       "
        # ANN2 = "     XXX "
        # PAR1 = "XXX      "
        # PAR1 = "    XXXXX"

        cls.part = Part('Here is a random sentence for the benefit of your mamma')
        cls.entity = Entity(class_id=STUB_ENTITY_CLASS_ID, offset=10, text='random sentence', confidence=0)
        cls.part.annotations.append(cls.entity)
        cls.doc.parts['s1h1'] = cls.part

        # Apply through pipeline

        NLTKSplitter().split(cls.dataset)
        NLTK_TOKENIZER.tokenize(cls.dataset)

        nlp = get_spacy_nlp_english(load_parser=True)
        cls.parser = SpacyParser(nlp)
        cls.parser.parse(cls.dataset)
        # cls.part.percolate_tokens_to_entities()

        cls.sentence = cls.part.sentences[0]

    def as_string(self, tokens):
        return ' '.join(t.word for t in tokens)


    def test_prev_tokens(self):

        self.assertEqual("a", self.as_string(self.entity.prev_tokens(self.sentence, n=1, include_ent_first_token=False)))
        self.assertEqual("is a", self.as_string(self.entity.prev_tokens(self.sentence, n=2, include_ent_first_token=False)))
        self.assertEqual("Here is a", self.as_string(self.entity.prev_tokens(self.sentence, n=3, include_ent_first_token=False)))
        self.assertEqual("Here is a", self.as_string(self.entity.prev_tokens(self.sentence, n=10, include_ent_first_token=False)))

        self.assertEqual("a random", self.as_string(self.entity.prev_tokens(self.sentence, n=1, include_ent_first_token=True)))
        self.assertEqual("is a random", self.as_string(self.entity.prev_tokens(self.sentence, n=2, include_ent_first_token=True)))
        self.assertEqual("Here is a random", self.as_string(self.entity.prev_tokens(self.sentence, n=3, include_ent_first_token=True)))
        self.assertEqual("Here is a random", self.as_string(self.entity.prev_tokens(self.sentence, n=10, include_ent_first_token=True)))


    def test_next_tokens(self):

        self.assertEqual("for", self.as_string(self.entity.next_tokens(self.sentence, n=1, include_ent_last_token=False)))
        self.assertEqual("for the", self.as_string(self.entity.next_tokens(self.sentence, n=2, include_ent_last_token=False)))
        self.assertEqual("for the benefit", self.as_string(self.entity.next_tokens(self.sentence, n=3, include_ent_last_token=False)))
        self.assertEqual("for the benefit of your mamma", self.as_string(self.entity.next_tokens(self.sentence, n=10, include_ent_last_token=False)))

        self.assertEqual("sentence for", self.as_string(self.entity.next_tokens(self.sentence, n=1, include_ent_last_token=True)))
        self.assertEqual("sentence for the", self.as_string(self.entity.next_tokens(self.sentence, n=2, include_ent_last_token=True)))
        self.assertEqual("sentence for the benefit", self.as_string(self.entity.next_tokens(self.sentence, n=3, include_ent_last_token=True)))
        self.assertEqual("sentence for the benefit of your mamma", self.as_string(self.entity.next_tokens(self.sentence, n=10, include_ent_last_token=True)))


    def test_overlapping(self):

        e1 = Entity(class_id="e_x", offset=987, text="PKB/Akt")
        e2 = Entity(class_id="e_x", offset=987, text="PKB")

        Entity.equality_operator = 'exact_or_overlapping'

        print(e1.offset, e1.end_offset())
        print(e2.offset, e2.end_offset())

        self.assertEqual(e1, e2)


class TestLabel(unittest.TestCase):
    pass


# @attr('slow')
class TestPart(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        nlp = get_spacy_nlp_english(load_parser=True)
        cls.parser = SpacyParser(nlp)

    def test_init(self):
        pass  # TODO

    def test_iter(self):
        pass  # TODO


    def _get_test_data(self, entity_sentence, assumed_tokens_words=None):
        if assumed_tokens_words is None:
            assumed_tokens_words = entity_sentence.split(' ')

        # Create dataset

        dataset = StringReader(entity_sentence).read()
        part = next(dataset.parts())
        entity = Entity(class_id=STUB_ENTITY_CLASS_ID, offset=0, text=entity_sentence)
        part.annotations.append(entity)

        # Apply through pipeline

        NLTKSplitter().split(dataset)
        NLTK_TOKENIZER.tokenize(dataset)
        self.parser.parse(dataset)

        # Rest

        sentences = part.sentences
        assert len(sentences) == 1
        sentence = sentences[0]

        assert len(assumed_tokens_words) == len(sentence)
        for (assumed_token_word, actual_token) in zip(assumed_tokens_words, sentence):
            assert assumed_token_word == actual_token.word

        part.compute_tokens_depth()
        roots = Part.get_sentence_roots(sentence)
        for r in roots:
            self._assert_depth_eq(r, 0)

        part.set_entities_head_tokens()

        return (dataset, sentence, entity, roots)


    def _assert_depth_eq(self, token, depth):
        assert token.features[Part._FEAT_DEPTH_KEY] == depth, (token, " : ", token.features[Part._FEAT_DEPTH_KEY], " != ", depth)


    def assert_depth_eq(self, sentence, word, depth):
        self._assert_depth_eq(next(filter(lambda t: t.word == word, sentence)), depth)


    def test_depths_and_head_token__the_root(self):
        # Deps graph: https://demos.explosion.ai/displacy/?text=ecto%20-%20nucleotide%20pyrophosphatase%20%2F%20phosphodiesterase%20I-1&model=en&cpu=0&cph=0
        (d, s, e, rs) = self._get_test_data('ecto - nucleotide pyrophosphatase / phosphodiesterase I-1')
        assert len(rs) == 1 and rs[0].word == 'phosphodiesterase'

        self.assert_depth_eq(s, 'ecto', 2)
        self.assert_depth_eq(s, '-', 2)
        self.assert_depth_eq(s, 'nucleotide', 1)
        self.assert_depth_eq(s, 'pyrophosphatase', 1)
        self.assert_depth_eq(s, '/', 1)
        self.assert_depth_eq(s, 'I-1', 1)

        assert e.head_token.word == 'phosphodiesterase', e.head_token.word


class TestMentionLevel(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        # create a sample dataset to test
        cls.dataset = Dataset()
        part = Part('some text c.A100G p.V100Q some text')
        part.sentences = [[Token('some', 0), Token('text', 5), Token('c', 10), Token('.', 11), Token('A', 12),
                           Token('100', 13), Token('G', 16), Token('p', 18), Token('.', 19), Token('V', 20),
                           Token('100', 21), Token('Q', 24), Token('some', 26), Token('text', 31)]]

        predicted_labels = ['O', 'O', 'B', 'I', 'I', 'I', 'E', 'A', 'I', 'I', 'I', 'E', 'O', 'O']

        for index, label in enumerate(predicted_labels):
            part.sentences[0][index].predicted_labels = [Label(label)]

        cls.dataset.documents['doc_1'] = Document()
        cls.dataset.documents['doc_1'].parts['p1'] = part

        part = Part('test edge case DNA A927B test')
        part.sentences = [[Token('test', 0), Token('edge', 5), Token('case', 10), Token('DNA', 15),
                           Token('A', 19), Token('927', 20), Token('B', 23), Token('test', 25)]]

        predicted_labels = ['O', 'O', 'O', 'O', 'M', 'P', 'M', 'O']

        for index, label in enumerate(predicted_labels):
            part.sentences[0][index].predicted_labels = [Label(label)]

        cls.dataset.documents['doc_1'].parts['p2'] = part

    def test_form_predicted_annotations(self):
        self.dataset.form_predicted_annotations(STUB_ENTITY_CLASS_ID)

        part = self.dataset.documents['doc_1'].parts['p1']

        self.assertEqual(len(part.predicted_annotations), 2)

        self.assertEqual(part.predicted_annotations[0].text, 'c.A100G')
        self.assertEqual(part.predicted_annotations[0].offset, 10)

        self.assertEqual(part.predicted_annotations[1].text, 'p.V100Q')
        self.assertEqual(part.predicted_annotations[1].offset, 18)

        part = self.dataset.documents['doc_1'].parts['p2']

        self.assertEqual(len(part.predicted_annotations), 1)

        self.assertEqual(part.predicted_annotations[0].text, 'A927B')
        self.assertEqual(part.predicted_annotations[0].offset, 19)


if __name__ == '__main__':
    unittest.main()
