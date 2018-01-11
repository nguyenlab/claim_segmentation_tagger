# -*- coding: utf-8 -*-
__author__ = "Danilo S. Carvalho <danilo@jaist.ac.jp>"

import json

config_cache = None


def load_config(path, cached=True):
    global config_cache

    if (cached and config_cache is not None):
        return config_cache

    config = {}
    with open(path) as config_file:
        config = json.load(config_file)

    return config


def update_config(config):
    global config_cache

    config_cache = config

