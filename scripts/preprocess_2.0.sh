#!/usr/bin/env bash

set -e

# ############### AMR v2.0 ################
# # Directory where intermediate utils will be saved to speed up processing.
util_dir=data/AMR/amr_2.0_utils

# AMR data with **features**
data_dir=data/AMR/amr_2.0
train_data=${data_dir}/train.txt.features
dev_data=${data_dir}/dev.txt.features
test_data=${data_dir}/test.txt.features

# ========== Set the above variables correctly ==========
printf "Cleaning inputs...`date`\n"
python -u -m stog.data.dataset_readers.amr_parsing.preprocess.input_cleaner \
    --amr_files ${train_data} ${dev_data} ${test_data}
printf "Done.`date`\n\n"

printf "Recategorizing subgraphs...`date`\n"
python -u -m stog.data.dataset_readers.amr_parsing.preprocess.recategorizer \
    --dump_dir ${util_dir} \
    --amr_files ${train_data}.input_clean ${dev_data}.input_clean
python -u -m stog.data.dataset_readers.amr_parsing.preprocess.graph_anonymizor \
    --amr_file ${test_data}.input_clean \
    --util_dir ${util_dir}
printf "Done.`date`\n\n"

printf "Removing senses...`date`\n"
python -u -m stog.data.dataset_readers.amr_parsing.preprocess.sense_remover \
    --util_dir ${util_dir} \
    --amr_files ${train_data}.input_clean.recategorize \
    ${dev_data}.input_clean.recategorize \
    ${test_data}.input_clean.recategorize
printf "Done.`date`\n\n"

printf "Renaming preprocessed files...`date`\n"
mv ${test_data}.input_clean.recategorize.nosense ${test_data}.preproc
mv ${train_data}.input_clean.recategorize.nosense ${train_data}.preproc
mv ${dev_data}.input_clean.recategorize.nosense ${dev_data}.preproc
rm ${data_dir}/*.recategorize*
