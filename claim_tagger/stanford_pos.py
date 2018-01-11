__author__ = "Danilo S. Carvalho <danilo@jaist.ac.jp>"

import re
import nltk
from nltk.tag.stanford import CoreNLPPOSTagger

from saf.annotators import Annotator, AnnotationError
from config import DEFAULT_CONFIG_PATH
from config.loader import load_config

class StanfordPOSAnnotator(Annotator):
    def __init__(self, config_path=DEFAULT_CONFIG_PATH):
        self.config = load_config(config_path)
        corenlp_config = self.config["data"]["stanford_corenlp"]
        self.tagger = CoreNLPPOSTagger(url="http://%s:%d" % (corenlp_config["host"], corenlp_config["port"]))

        self.pos_map = self.config["model"]["STANFORD_POS_MAP"]

    def annotate(self, annotable):
        if (annotable.__class__.__name__ == "Document"):
            return self.annotate_document(annotable)
        elif (annotable.__class__.__name__ == "Sentence"):
            return self.annotate_sentence(annotable)
        else:
            raise AnnotationError("This annotator only accepts Document or Sentence annotables.")

    def annotate_document(self, document):
        for sentence in document.sentences:
            self.annotate_sentence(sentence)

    def annotate_sentence(self, sentence):
        token_list = [token.surface for token in sentence.tokens]
        tagged_tokens = self.tagger.tag(token_list)

        for i in range(len(token_list)):
            sentence.tokens[i].annotations["STANFORD_POS"] = tagged_tokens[i][1]

            for pos_rgx in self.pos_map:
                if (re.match(pos_rgx, tagged_tokens[i][1])):
                    sentence.tokens[i].annotations["POS"] = self.pos_map[pos_rgx].split("|")[0]

            if ("POS" not in sentence.tokens[i].annotations):
                sentence.tokens[i].annotations["POS"] = "x"
