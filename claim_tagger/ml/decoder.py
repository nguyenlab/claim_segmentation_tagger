__author__ = "Danilo S. Carvalho <danilo@jaist.ac.jp>"

import numpy as np


def decode_sentence(enc_sent_segments, classes):
    return [classes[np.argmax(enc_seg)] for enc_seg in enc_sent_segments]


def confidence(enc_sent_segments):
    return [float(np.nanmax(enc_seg)) for enc_seg in enc_sent_segments]
