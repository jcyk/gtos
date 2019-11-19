dataset=../data/AMR/amr_2.0
python3 work.py --test_data ${dataset}/test.txt.features.preproc.json \
               --test_batch_size 2000 \
               --load_path ./amr_ckpt/amr2/epoch624_batch119999 \
               --beam_size 8\
               --alpha 0.6\
               --max_time_step 100\
               --output_suffix _test_out
