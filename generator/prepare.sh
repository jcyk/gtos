dataset=../data/AMR/amr_2.0
python3 extract.py --train_data ${dataset}/train.txt.features.preproc \
                   --amr_files ${dataset}/train.txt.features.preproc ${dataset}/dev.txt.features.preproc ${dataset}/test.txt.features.preproc \
                   --nprocessors 8
mv *_vocab ${dataset}/.
