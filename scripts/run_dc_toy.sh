#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

PYTHON=python

TASK=humanoid
POP_SIZE=2560
NUM_RUNNERS=4
LR=0.15
CR_ONLY=false
CROSSOVER_METHOD="random-pick" # arithmetic, random-pick, logit-space, sample-based
CROSSOVER_RATIO=2.0

$PYTHON dc-toy-simplified.py \
    task=$TASK \
    cr_only=$CR_ONLY \
    es_conf.pop_size=$POP_SIZE \
    es_conf.num_runners=$NUM_RUNNERS \
    es_conf.lr=$LR \
    es_conf.crossover_method=$CROSSOVER_METHOD \
    es_conf.crossover_ratio=$CROSSOVER_RATIO

