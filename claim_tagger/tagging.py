# -*- coding: utf-8 -*-
__author__ = "Danilo S. Carvalho <danilo@jaist.ac.jp>"

import sys
import json
import re
import numpy as np
import logging
from itertools import izip
from collections import Counter
from joblib import Parallel, delayed
from saf.importers import CoNLLImporter
from saf.constants import annotation

from ml.encoder import encode_document
from ml.decoder import decode_sentence, confidence
from ml.deep_tagger import DeepClaimTagger
from config.loader import load_config

logger = logging.getLogger(__name__)

model_cache = None


class EnsembleMode:
    MAJORITY_OVERALL = 1
    MAJORITY_TOKEN = 2
    CONFIDENCE_OVERALL = 3
    CONFIDENCE_TOKEN = 4


def tag_claims(doc, config_path, model=None, model_path=None, cache=True):
    global model_cache
    config = load_config(config_path)
    classes = sorted(config["model"]["CLASSES"])
    annotation_types = config["model"]["ANNOTATIONS"]

    if (model is None):
        if (not model_path):
            if (cache and model_cache is not None):
                model = model_cache
            else:
                model = DeepClaimTagger(config)
                model.load(config["data"]["model_path"] % (config["id"], 0))
                model_cache = model
        else:
            model = DeepClaimTagger(config)
            model.load(model_path)
            model_cache = model

    sent_encs = encode_document(doc, config)[0]
    input_morph = []
    input_attr = []
    for (enc_morph, enc_attrs) in sent_encs:
        input_morph.append(enc_morph)
        input_attr.append(enc_attrs)

    enc_sent_segments = model.predict([np.array(input_morph, dtype=np.uint8), np.array(input_attr, dtype=np.float32)])

    for (sentence, enc_sent_seg) in izip(doc.sentences, enc_sent_segments):
        dec_segments = decode_sentence(enc_sent_seg, classes)
        token_confidence = confidence(enc_sent_seg)
        avg_confidence = [float(np.mean(token_conf)) for token_conf in token_confidence]

        for i in xrange(len(sentence.tokens)):
            segcls = dec_segments[i]
            sentence.tokens[i].annotations["PATCLAIM_SEG"] = segcls

        sentence.annotations["PATCLAIM_SEG_INFO"] = {"token_confidence": token_confidence, "confidence": avg_confidence}

        logger.info("Segmentation: " + " ".join([token.surface + ":" + seg for (token, seg) in izip(sentence.tokens, dec_segments[0:len(sentence.tokens)])]))

    return doc


def train_model(train_doc, config, model_seq=0, num_epochs=15, batch_size=20):
    input_seqs, output_seqs, sample_weights = encode_document(train_doc, config, training=True)

    input_morph = []
    input_attr = []
    for (enc_morph, enc_attrs) in input_seqs:
        input_morph.append(enc_morph)
        input_attr.append(enc_attrs)

    tagger = DeepClaimTagger(config)

    tagger.train([np.array(input_morph, dtype=np.uint8), np.array(input_attr, dtype=np.float32)],
                 np.array(output_seqs, dtype=np.uint8), num_epochs, batch_size=batch_size,  # validation_split=0.1,
                 sample_weight=np.array(sample_weights, dtype=np.uint8))
    tagger.save(config["data"]["model_path"] % (config["id"], model_seq))

    return tagger


def model_paths(config_paths=[]):
    model_num = 0
    last_config_path = ""
    paths = []
    for config_path in config_paths:
        if (config_path == last_config_path):
            model_num += 1
        else:
            model_num = 0

        config = load_config(config_path)
        paths.append(config["data"]["model_path"] % (config["id"], model_num))
        last_config_path = config_path

    return paths


def main(argv):
    config = load_config(argv[1])
    ops = argv[2].split(",")
    model = None

    if ("train" in ops):
        for i in xrange(5):
            model = train_model(config, model_seq=i+3)

    # if (model is None):
    #     model = Word2Morpho(config)
    #     model.load(config["data"]["model_path"] % (config["id"], 0))




if __name__ == "__main__":
    main(sys.argv)
