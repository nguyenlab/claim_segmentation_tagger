# -*- coding: utf-8 -*-
__author__ = "Danilo S. Carvalho <danilo@jaist.ac.jp>"


import unittest
import os
import cPickle
import json
import copy
import numpy as np
import logging
from collections import Counter
from saf.constants import annotation
from saf import Document
from saf.importers import WebAnnoImporter

from wikt_morphodecomp.annotation import MorphoAnalysisAnnotator
from tdv_postagger.annotation import POSAnnotator
from claim_tagger.stanford_pos import StanfordPOSAnnotator
from claim_tagger.annotation import SemVectorAnnotator, ClaimAnnotator
from claim_tagger.annotation import set_boundaries, combine_claimseg_classes, get_claimseg_classes

from claim_tagger.tagging import train_model
from claim_tagger.config.loader import load_config, update_config

logging.basicConfig(filename="test_segmentation.log", format='%(levelname)s:%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


EPO_SAMPLE_PATH = "./data/epo_annot_sample"
CONFIG_FILEPATH = "./data/config/default.config.json"
W2M_CONFIG_FILEPATH = "./data/config/w2m/wikt_morphodecomp.config.json"
POS_CONFIG_FILEPATH = "./data/config/pos/pos_tagger.config.json"


def normalize(v):
    norm = np.linalg.norm(v)
    return v if (norm == 0) else v / norm


def evaluate(test_set, gold_set, segment_classes):
    doc_precisions = [0.0] * len(test_set)
    doc_recalls = [0.0] * len(test_set)
    doc_fscores = [0.0] * len(test_set)
    for i in range(len(test_set)):
        counter_init = {cls: 0 for cls in segment_classes}
        class_hits = Counter(counter_init)
        test_totals = Counter(counter_init)
        gold_totals = Counter(counter_init)
        precision = dict(counter_init)
        recall = dict(counter_init)
        doc_precisions[i] = 0.0
        doc_recalls[i] = 0.0

        for j in range(len(test_set[i].sentences)):
            for k in range(len(test_set[i].sentences[j].tokens)):
                gold_label = gold_set[i].sentences[j].tokens[k].annotations["PATCLAIM_SEG"]
                test_label = test_set[i].sentences[j].tokens[k].annotations["PATCLAIM_SEG"]
                if (test_label == gold_label):
                    class_hits[gold_label] += 1
                test_totals[test_label] += 1
                gold_totals[gold_label] += 1

        for seg_class in counter_init:
            if (gold_totals[seg_class] > 0):
                precision[seg_class] = 0.0 if (test_totals[seg_class] == 0) else float(class_hits[seg_class]) / test_totals[seg_class]
                recall[seg_class] = float(class_hits[seg_class]) / gold_totals[seg_class]

        for seg_class in counter_init:
            if (gold_totals[seg_class] == 0):
                del precision[seg_class]
                del recall[seg_class]

        total_segs = sum(gold_totals.values())
        weights = {seg_class: float(gold_totals[seg_class]) / total_segs for seg_class in gold_totals}

        indexed_precision = []
        indexed_recall = []
        indexed_weights = []
        for seg_class in precision:
            indexed_precision.append(precision[seg_class])
            indexed_recall.append(recall[seg_class])
            indexed_weights.append(weights[seg_class])

        doc_precisions[i] = np.average(indexed_precision, weights=normalize(np.array(indexed_weights)))
        doc_recalls[i] = np.average(indexed_recall, weights=normalize(np.array(indexed_weights)))
        doc_fscores[i] = (2 * doc_precisions[i] * doc_recalls[i]) / (doc_precisions[i] + doc_recalls[i])

        test_set[i].annotations["PATCLAIM_SEG_TEST_RESULTS"] = {"precision_per_class": precision, "recall_per_class": recall,
                                                                "precision": doc_precisions[i], "recall": doc_recalls[i],
                                                                "f-score": doc_fscores[i], "weights": normalize(np.array(indexed_weights)).tolist()}
        logger.info("Document %s results:")
        logger.info("%s", json.dumps(test_set[i].annotations["PATCLAIM_SEG_TEST_RESULTS"], indent=2))

    return (np.average(doc_precisions), np.average(doc_recalls), np.average(doc_fscores))



class TestClaimTagger(unittest.TestCase):
    def test_training(self):
        config = load_config(CONFIG_FILEPATH)
        model = None
        annotation_types = set(config["model"]["ANNOTATIONS"])
        epo_sample_docs = []

        logger.info("Loading inputs...")
        webanno_importer = WebAnnoImporter(["PATCLAIM_SEG"])


        try:
            with open("./data/epo_experiment_corpus.pickle", "rb") as epo_annot_corpus_file:
                epo_sample_docs = cPickle.load(epo_annot_corpus_file)

        except IOError:
            for filename in os.listdir(EPO_SAMPLE_PATH):
                with open(os.path.join(EPO_SAMPLE_PATH, filename)) as epo_annot_file:
                    epo_sample_docs.append(webanno_importer.import_document(unicode(epo_annot_file.read(), encoding="utf8").strip()))
                    epo_sample_docs[-1].title = filename

            for epo_doc in epo_sample_docs:
                set_boundaries(epo_doc, config["model"]["DEPTH_OUTPUT"])
                combine_claimseg_classes(epo_doc)

            if ("MORPHO" in annotation_types):
                logger.info("Running morphological decomposition...")
                morpho_annotator = MorphoAnalysisAnnotator()
                for epo_doc in epo_sample_docs:
                    morpho_annotator.annotate(epo_doc, ensemble=True, config_paths=(W2M_CONFIG_FILEPATH,))

            if ("POS" in annotation_types):
                logger.info("Running POS tagging...")

                if (config["options"]["tagger"] == "stanford"):
                    postag_annotator = StanfordPOSAnnotator()
                    for epo_doc in epo_sample_docs:
                        postag_annotator.annotate(epo_doc)

                else:
                    postag_annotator = POSAnnotator()
                    for epo_doc in epo_sample_docs:
                        postag_annotator.annotate(epo_doc, config_paths=(POS_CONFIG_FILEPATH,))

            if ("W2V" in annotation_types or "TDV" in annotation_types):
                logger.info("Running semantic vector annotation (word2vec, TDV)...")
                semvector_annotator = SemVectorAnnotator()
                for epo_doc in epo_sample_docs:
                    semvector_annotator.annotate(epo_doc)

            with open("./data/epo_experiment_corpus.pickle", "wb") as epo_annot_corpus_file:
                cPickle.dump(epo_sample_docs, epo_annot_corpus_file, cPickle.HIGHEST_PROTOCOL)

        segment_classes = get_claimseg_classes(epo_sample_docs)
        config["model"]["CLASSES"] = segment_classes
        update_config(config)

        precision_folds = np.zeros(len(epo_sample_docs))
        recall_folds = np.zeros(len(epo_sample_docs))
        fscore_folds = np.zeros(len(epo_sample_docs))

        for loo_idx in range(len(epo_sample_docs)):
            training_set = []
            test_set = []
            gold_set = []

            for i in range(len(epo_sample_docs)):
                if (i == loo_idx):
                    test_set.append(copy.deepcopy(epo_sample_docs[i]))
                    gold_set.append(epo_sample_docs[i])
                else:
                    training_set.append(epo_sample_docs[i])

            for test_doc in test_set:
                for sentences in test_doc.sentences:
                    for token in sentences.tokens:
                        del token.annotations["PATCLAIM_SEG"]

            combined_training_doc = Document()
            for epo_doc in training_set:
                combined_training_doc.sentences.extend(epo_doc.sentences)

            logger.info("Using annotations: %s", ", ".join(annotation_types))

            logger.info("Training claim segmentation model...")
            model = train_model(combined_training_doc, config, model_seq=0)

            claim_annotator = ClaimAnnotator()

            logger.info("Annotating test claims...")
            for epo_doc in test_set:
                claim_annotator.annotate(epo_doc, model=model)

            logger.info("Evaluating performance: ")
            precision, recall, fscore = evaluate(test_set, gold_set, segment_classes)
            precision_folds[loo_idx] = precision
            recall_folds[loo_idx] = recall
            fscore_folds[loo_idx] = fscore
            with open("./data/epo_experiment_results_fold[%d].pickle" % loo_idx, "wb") as epo_result_file:
                cPickle.dump(test_set, epo_result_file, cPickle.HIGHEST_PROTOCOL)

            logger.info("Precision fold (%d) [%s]: %.2f, Recall: %.2f, F-score: %.2f", loo_idx, test_set[0].title, precision, recall, fscore)

        logger.info("Precision: %.2f, Recall: %.2f, F-score: %.2f",
                    np.average(precision_folds), np.average(recall_folds), np.average(fscore_folds))




