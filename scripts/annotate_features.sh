#!/usr/bin/env bash

set -e

# Start a Stanford CoreNLP server before running this script.
# https://stanfordnlp.github.io/CoreNLP/corenlp-server.html

# The compound file is downloaded from
# https://github.com/ChunchuanLv/AMR_AS_GRAPH_PREDICTION/blob/master/data/joints.txt
compound_file=data/AMR/amr_2.0_utils/joints.txt
amr_dir=$1

python -u -m stog.data.dataset_readers.amr_parsing.preprocess.feature_annotator \
    ${amr_dir}/test.txt ${amr_dir}/train.txt ${amr_dir}/dev.txt \
    --compound_file ${compound_file}