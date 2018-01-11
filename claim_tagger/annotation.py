__author__ = "Danilo S. Carvalho <danilo@jaist.ac.jp>"

import cPickle
import numpy as np
import logging
import nltk
from nltk.tag.stanford import CoreNLPPOSTagger
from pymongo import MongoClient
from saf import Token
from saf.annotators import Annotator, AnnotationError
# from tagging import load_models, EnsembleMode
from tagging import tag_claims
from config import DEFAULT_CONFIG_PATH
from config.loader import load_config

logger = logging.getLogger(__name__)


def set_boundaries(document, depth):
    for sentence in document.sentences:
        last_label = [""] * depth
        last_label_count = [0] * depth
        last_token = None
        for token in sentence.tokens:
            for i in range(len(token.annotations["PATCLAIM_SEG"][0:depth])):
                label = token.annotations["PATCLAIM_SEG"][i]
                if (label != last_label[i]):
                    token.annotations["PATCLAIM_SEG"][i] = token.annotations["PATCLAIM_SEG"][i] + "_B"

                    if (last_token is not None and i < len(last_token.annotations["PATCLAIM_SEG"])):
                        if (len(last_token.annotations["PATCLAIM_SEG"]) > len(token.annotations["PATCLAIM_SEG"])):
                            for j in range(len(last_token.annotations["PATCLAIM_SEG"]) - len(token.annotations["PATCLAIM_SEG"])):
                                if (last_label_count[0:len(token.annotations["PATCLAIM_SEG"])][-j] > 1):
                                    last_token.annotations["PATCLAIM_SEG"][0:len(token.annotations["PATCLAIM_SEG"])][-j] = last_label[-j] + "_E"
                                else:
                                    last_token.annotations["PATCLAIM_SEG"][0:len(token.annotations["PATCLAIM_SEG"])][-j] = last_label[-j] + "_S"
                        else:
                            if (last_label_count[i] > 1):
                                last_token.annotations["PATCLAIM_SEG"][i] = last_label[i] + "_E"
                            else:
                                last_token.annotations["PATCLAIM_SEG"][i] = last_label[i] + "_S"

                    last_label_count[i] = 1
                else:
                    token.annotations["PATCLAIM_SEG"][i] = token.annotations["PATCLAIM_SEG"][i] + "_M"
                    last_label_count[i] += 1

                last_label[i] = label
            last_token = token


def combine_claimseg_classes(document):
    for sentence in document.sentences:
        for token in sentence.tokens:
            token.annotations["PATCLAIM_SEG"] = "->".join(token.annotations["PATCLAIM_SEG"])


def get_claimseg_classes(documents):
    classes = set()
    for document in documents:
        for sentence in document.sentences:
            for token in sentence.tokens:
                classes.add(token.annotations["PATCLAIM_SEG"])

    classes.add(":end:")

    return list(classes)


def get_segments(doc, config_paths, ensemble, model=None):
    if (ensemble):
        print "Not implemented."
        # models = load_models(config_paths)
        # tagged_doc = postag_ensemble(doc, config_paths, models=models, mode=EnsembleMode.MAJORITY_OVERALL)
    else:
        tagged_doc = tag_claims(doc, config_paths[0], model=model, cache=False)

    return tagged_doc


class ClaimAnnotator(Annotator):
    def annotate(self, annotable, ensemble=False, config_paths=(DEFAULT_CONFIG_PATH,), model=None):
        if (annotable.__class__.__name__ == "Document"):
            return ClaimAnnotator.annotate_document(annotable, ensemble, config_paths, model=model)
        else:
            raise AnnotationError("This annotator only accepts document annotables.")

    @staticmethod
    def annotate_document(document, ensemble, config_paths, model=None):
        return get_segments(document, config_paths, ensemble, model=model)


class SemVectorAnnotator(Annotator):
    def __init__(self, config_path=DEFAULT_CONFIG_PATH):
        self.config = load_config(config_path)
        self.conn = MongoClient(self.config["data"]["semantic_db"]["host"], self.config["data"]["semantic_db"]["port"])
        self.db = self.conn.semdb
        self.pos_map = self.config["model"]["POS_MAP"]

    @staticmethod
    def combine_tdvs(tdv_list):
        combined_tdv = dict()
        for tdv in tdv_list:
            for link in tdv:
                link_id = link["term"] + "|" + str(link["type_id"])
                if (link_id not in combined_tdv):
                    combined_tdv[link_id] = link
                else:
                    combined_tdv[link_id]["value"] += link["value"]

        for link_id in combined_tdv:
            combined_tdv[link_id]["value"] /= len(tdv_list)

        return combined_tdv.values()

    def get_tdv_links(self, token, tdv):
        pos = "x"
        if ("POS" in token.annotations):
            pos = token.annotations["POS"]

        if (pos in tdv["meanings"]):
            links = tdv["meanings"][pos]
        else:
            links = SemVectorAnnotator.combine_tdvs(tdv["meanings"].values())

        return links

    def gen_tdv_attrs(self, token, links, w2vdb):
        ranked_links = sorted(links, key=lambda x: x["value"], reverse=True)[0:self.config["model"]["TDV_SIZE"]]
        token.annotations["TDV"] = sorted(ranked_links, key=lambda x: x["term"])
        max_val = max([link["value"] for link in ranked_links])
        w2v_dim = self.config["model"]["W2V_DIM"]
        tdv_size = self.config["model"]["TDV_SIZE"]
        tdv_w2v = np.zeros((tdv_size, w2v_dim), dtype=np.float32)
        for link in token.annotations["TDV"]:
            link["value"] /= max_val

        for i in xrange(len(token.annotations["TDV"])):
            link = token.annotations["TDV"][i]
            w2v = w2vdb.find_one({"word": link["term"]})

            if (w2v is not None):
                tdv_w2v[i] = cPickle.loads(w2v["vector"]) * float(link["value"])

        token.annotations["TDV-W2V"] = tdv_w2v.flatten()

    def annotate(self, annotable):
        if (annotable.__class__.__name__ == "Document"):
            return self.annotate_document(annotable)
        else:
            raise AnnotationError("This annotator only accepts document annotables.")

    def annotate_document(self, document):
        for sentence in document.sentences:
            for token in sentence.tokens:
                tdv = self.db.tdv.find_one({"term": token.surface})

                if (tdv is not None):
                    links = self.get_tdv_links(token, tdv)
                    self.gen_tdv_attrs(token, links, self.db.w2v)

                else:
                    logger.info("Word not found in TDV vocabulary: " + token.surface)
                    if ("MORPHO" in token.annotations):
                        morpheme_tdvs = []
                        for morpheme in token.annotations["MORPHO"]["decomp"]:
                            tdv_morph = self.db.tdv.find_one({"term": morpheme})

                            if (tdv_morph is not None):
                                morph_token = Token()
                                morph_token.surface = morpheme
                                morpheme_tdvs.append(self.get_tdv_links(morph_token, tdv_morph))

                        combined_morph_tdv_links = SemVectorAnnotator.combine_tdvs(morpheme_tdvs)

                        if (len(combined_morph_tdv_links) > 0):
                            self.gen_tdv_attrs(token, combined_morph_tdv_links, self.db.w2v_decomp)
                            logger.info("Successful morphological decomposition into TDV: " + " ".join(token.annotations["MORPHO"]["decomp"]))
                        else:
                            token.annotations["TDV"] = []
                            token.annotations["TDV-W2V"] = np.zeros(self.config["model"]["W2V_DIM"] * self.config["model"]["TDV_SIZE"])
                            logger.info("Unsuccessful morphological decomposition into TDV: " + " ".join(token.annotations["MORPHO"]["decomp"]))
                    else:
                        token.annotations["TDV"] = []
                        token.annotations["TDV-W2V"] = np.zeros(self.config["model"]["W2V_DIM"] * self.config["model"]["TDV_SIZE"])

                w2v = self.db.w2v.find_one({"word": token.surface})

                if (w2v is not None):
                    token.annotations["W2V"] = cPickle.loads(w2v["vector"])
                else:
                    token.annotations["W2V"] = np.zeros(self.config["model"]["W2V_DIM"])

