from textblob import TextBlob
from textblob.en.np_extractors import FastNPExtractor

from nala.features import FeatureGenerator


class PosTagFeatureGenerator(FeatureGenerator):
    """
    """

    def __init__(self):
        self.punctuation = ['.', ',', ':', ';', '[', ']', '(', ')', '{', '}', '”', '“', '–', '"', '#', '?', '-']

    def generate(self, dataset):
        """
        :type dataset: nala.structures.data.Dataset
        """
        for part in dataset.parts():
            tags = TextBlob(part.text).tags

            tag_index = 0
            for sentence in part.sentences:
                for token in sentence:
                    if token.word in self.punctuation:
                        token.features['tag'] = 'PUN'
                    else:
                        remember_index = tag_index
                        for word, tag in tags[tag_index:]:
                            if token.word in word:
                                token.features['tag'] = tag
                                tag_index += 1
                                break
                        tag_index = remember_index


class NounPhraseFeatureGenerator(FeatureGenerator):
    """

    """
    def __init__(self):
        self.extractor = FastNPExtractor()

    def generate(self, dataset):
        """
        :type dataset: nala.structures.data.Dataset
        """
        for part in dataset.parts():
            for sentence in part:
                # get the chunk of text representing the sentence
                joined_sentence = part.text[sentence[0].start:sentence[-1].start + len(sentence[-1].word)]
                phrases = self.extractor.extract(joined_sentence)
                for phrase in phrases:
                    # only consider real noun phrases that have more than 1 word
                    if ' ' in phrase:
                        # find the phrase offset in part text
                        phrase_start = part.text.find(phrase, sentence[0].start)
                        phrase_end = phrase_start + len(phrase)

                        # mark the tokens that are part of that phrase
                        for token in sentence:
                            if phrase_start <= token.start < token.end <= phrase_end:
                                token.features['is_nn'] = 1
