from nalaf.features import FeatureGenerator
from nalaf.features import get_spacy_nlp_english


class SpacyPosTagger(FeatureGenerator):
    """
    POS-tag a dataset using the Spacy Pos Tagger
    """

    def __init__(self):
        self.nlp = get_spacy_nlp_english()

    def generate(self, dataset):
        """
        :type dataset: nalaf.structures.data.Dataset
        """

        for part in dataset.parts():
            for sentence in part.sentences:
                text_tokens = list(map(lambda x: x.word, sentence))
                spacy_doc = self.nlp.tokenizer.tokens_from_list(text_tokens)

                self.nlp.tagger(spacy_doc)
                for token, spacy_token in zip(sentence, spacy_doc):
                    token.features['pos'] = spacy_token.pos_
                    token.features['tag'] = spacy_token.tag_
