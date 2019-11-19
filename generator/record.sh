dataset=../data/AMR/amr_2.0
python3 record.py --test_data ${dataset}/dev.txt.features.preproc.json \
               --test_batch_size 2000 \
               --load_path ./amr_ckpt/amr2/epoch624_batch119999 \
               --output_suffix _record_out
