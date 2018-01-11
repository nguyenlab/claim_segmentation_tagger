#-*- coding: utf8 -*-
__author__ = "Danilo S. Carvalho <danilo@jaist.ac.jp>"

import numpy as np
import string

CHAR_CODES = dict([(unicode(c), idx) for (idx, c) in enumerate(list(string.ascii_lowercase) + ['\'', '-', '{', '}', ' ', chr(1)])])
ENC_SIZE_CHAR = len(CHAR_CODES)

pos_db = dict()


def encode_char(c, char_size=ENC_SIZE_CHAR):
    enc = np.zeros(char_size, dtype=np.uint8)

    if (c.lower() in CHAR_CODES):
        c_code = CHAR_CODES[c.lower()]
        enc[c_code] = 1
    else:
        enc[CHAR_CODES[unichr(1)]] = 1

    return enc


def encode_token(w, word_size, char_size, char_win_size, reverse=False):
    enc = np.zeros((word_size, char_size * char_win_size), dtype=np.uint8)

    assert (char_win_size % 2) == 1
    assert char_win_size >= 1

    if (len(w) > word_size):
        print "Warning: Token exceeds maximum length: %d. Will be truncated." % word_size
        print w
        print "Length: ", len(w)

    w = w[0:word_size - 2]

    if (not reverse):
        charseq = list("{" + w + "}")
    else:
        charseq = list(u"{" + w[::-1] + u"}")

    lpadded = char_win_size // 2 * [u"{"] + charseq + char_win_size // 2 * [u"}"]
    context_windows = [lpadded[i:(i + char_win_size)] for i in range(len(charseq))]

    for i in xrange(len(context_windows)):
        enc[i] = np.concatenate([encode_char(c) for c in context_windows[i]])

    return enc


def encode_attributes(token, annotation_types, w2v_dim, tdv_link_dim, tdv_size, pos_map, pos_idx):
    enc_pos = np.zeros(len(pos_idx), dtype=np.float32)
    enc_tdv_linktypes = np.zeros((tdv_size, tdv_link_dim), dtype=np.float32)
    enc_tdv_highest = np.zeros(tdv_size * w2v_dim, dtype=np.float32)
    w2v = np.zeros(w2v_dim, dtype=np.float32)

    if ("POS" in token.annotations):
        pos = token.annotations["POS"]
        enc_pos[pos_idx[pos]] = 1.0

    if ("W2V" in token.annotations):
        w2v = token.annotations["W2V"]

    if ("TDV" in token.annotations):
        enc_tdv_highest = token.annotations["TDV-W2V"]

        for i in xrange(len(token.annotations["TDV"])):
            enc_tdv_linktypes[i][token.annotations["TDV"][i]["type_id"]] = 1.0

    attr_list = []

    if ("POS" in annotation_types):
        attr_list.append(enc_pos)

    if ("W2V" in annotation_types):
        attr_list.append(w2v)

    if ("TDV" in annotation_types):
        attr_list.append(enc_tdv_highest)
        attr_list.append(enc_tdv_linktypes.flatten())

    if (len(attr_list) > 0):
        return np.concatenate(attr_list)
    else:
        return np.zeros(1)


def encode_sentence(sent, max_sent_len, token_size, ctx_win_size, char_win_size, annotation_types,
                    w2v_dim, tdv_link_dim, tdv_size, pos_map, pos_idx):
    enc_token_windows = np.zeros((max_sent_len, token_size * ctx_win_size, ENC_SIZE_CHAR * char_win_size), dtype=np.uint8)

    enc_tokens = []
    enc_attrs = []

    if (len(sent.tokens) > max_sent_len):
        print "Warning: Sentence exceeds maximum length: %d. Will be truncated." % max_sent_len
        print [tok.surface for tok in sent.tokens]
        print "Length: ", len(sent.tokens)

    for i in xrange(len(sent.tokens[0:max_sent_len])):
        token = sent.tokens[i]
        if ("MORPHO" in token.annotations):
            enc_tokens.append(encode_token(" ".join(token.annotations["MORPHO"]["decomp"]), token_size, ENC_SIZE_CHAR,
                                           char_win_size))
        else:
            enc_tokens.append(encode_token(token.surface, token_size, ENC_SIZE_CHAR, char_win_size))
            print "Warning: Tokens without morphological annotations. Using surface forms only."

        enc_attrs.append(encode_attributes(token, annotation_types, w2v_dim, tdv_link_dim, tdv_size, pos_map, pos_idx))

    start_token = np.zeros((token_size, ENC_SIZE_CHAR * char_win_size), dtype=np.uint8)
    end_token = np.zeros((token_size, ENC_SIZE_CHAR * char_win_size), dtype=np.uint8)
    for i in xrange(token_size):
        start_token[i] = np.concatenate([encode_char(u"{")] * char_win_size)
        end_token[i] = np.concatenate([encode_char(u"}")] * char_win_size)

    attr_dim = enc_attrs[0].shape[0]
    enc_attr_windows = np.zeros((max_sent_len, attr_dim * ctx_win_size), dtype=np.float32)

    start_attr = np.zeros(attr_dim, dtype=np.uint8)
    end_attr = np.zeros(attr_dim, dtype=np.uint8)

    padded_token_seq = ctx_win_size // 2 * [start_token] + enc_tokens + ctx_win_size // 2 * [end_token]
    padded_attr_seq = ctx_win_size // 2 * [start_attr] + enc_attrs + ctx_win_size // 2 * [end_attr]

    padded_token_windows = [padded_token_seq[i:(i + ctx_win_size)] for i in range(len(enc_tokens))]
    padded_attr_windows = [padded_attr_seq[i:(i + ctx_win_size)] for i in range(len(enc_attrs))]

    for i in xrange(len(sent.tokens[0:max_sent_len])):
        enc_token_windows[i] = np.concatenate(padded_token_windows[i], axis=0)
        enc_attr_windows[i] = np.concatenate(padded_attr_windows[i], axis=0)

    return (enc_token_windows, enc_attr_windows)


def encode_segment_classes(sent, max_sent_len, class_idx, depth_out):
    enc = np.zeros((max_sent_len, 1), dtype=np.uint8)

    for i in xrange(len(sent.tokens[0:max_sent_len])):
        enc[i] = class_idx[sent.tokens[i].annotations["PATCLAIM_SEG"]]

    for i in xrange(len(sent.tokens), max_sent_len):
        enc[i] = class_idx[":end:"]

    return enc


def encode_sample_weights(sent, max_sent_len):
    enc = np.zeros(max_sent_len, dtype=np.uint8)

    for i in xrange(len(sent.tokens[0:max_sent_len])):
        enc[i] = 1

    return enc


def encode_document(doc, config, training=False):
    max_sent_len = config["model"]["MAX_SENT_LEN"] if (training) else config["model"]["MAX_SENT_LEN_TEST"]
    token_size = config["model"]["ENC_SIZE_TOKEN"]
    char_win_size = config["model"]["ENC_CHAR_SIZE_CTX_WIN"]
    ctx_win_size = config["model"]["ENC_SIZE_CTX_WIN"]
    annotation_types = set(config["model"]["ANNOTATIONS"])
    w2v_dim = config["model"]["W2V_DIM"]
    tdv_size = config["model"]["TDV_SIZE"]
    tdv_link_dim = config["model"]["TDV_LINK_DIM"]
    pos_idx = dict([(pos, idx) for (idx, pos) in list(enumerate(sorted(config["model"]["POS_CLASSES"])))])
    class_idx = dict([(cls, idx) for (idx, cls) in list(enumerate(sorted(config["model"]["CLASSES"])))])
    depth_out = config["model"]["DEPTH_OUTPUT"]
    pos_map = config["model"]["POS_MAP"]

    sent_encs = []
    segment_class_encs = []
    sample_weight_encs = []

    for sentence in doc.sentences:
        sent_encs.append(encode_sentence(sentence, max_sent_len, token_size, ctx_win_size, char_win_size,
                                         annotation_types, w2v_dim, tdv_link_dim, tdv_size, pos_map, pos_idx))

        if (training):
            segment_class_encs.append(encode_segment_classes(sentence, max_sent_len, class_idx, depth_out))
            sample_weight_encs.append(encode_sample_weights(sentence, max_sent_len))

    return sent_encs, segment_class_encs, sample_weight_encs



