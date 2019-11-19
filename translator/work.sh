dataset=../data/cs
python3 work.py --test_data ${dataset}/dev.txt \
               --test_batch_size 44444 \
               --load_path ckpt/epoch6_batch21999 \
               --beam_size 8\
               --alpha 0.6\
               --max_time_step 100\
               --output_suffix _test_out
