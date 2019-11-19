dataset=../data/de
python3 extract.py --train_data ${dataset}/train.txt \
                   --nprocessors 16
mv *_vocab ${dataset}/.
