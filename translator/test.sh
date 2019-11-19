for file in `ls ckpt/epoch6*_test_out*`
do
    echo ${file}
    python3 postprocess.py --golden_file ../data/cs/newstest2015-encs-ref.tok.cs \
                        --pred_file ${file} \
                        --output
done
