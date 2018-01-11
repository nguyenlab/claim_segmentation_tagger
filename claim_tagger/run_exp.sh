#!/usr/bin/env bash

source activate py27
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=..:./test:$PYTHONPATH #:/Users/danilo/research/tools:/Users/danilo/research/tools/saf:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0


python -m unittest test_segmentation.TestClaimTagger
