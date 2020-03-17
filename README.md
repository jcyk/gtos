# Graph Transformer

Code for our AAAI2020 paper,

Graph Transformer for Graph-to-Sequence Learning. [[preprint]](https://arxiv.org/pdf/1911.07470.pdf)

Deng Cai and Wai Lam.

## 1. Environment Setup

The code is tested with Python 3.6. All dependencies are listed in [requirements.txt](requirements.txt).

## 2. Data Preprocessing
The instructions for Syntax-based Machine Translation are given in the [translator_data](./translator_data) folder.

The instructions for AMR-to-Text Generation are given in the [generator_data](./generator_data) folder.

---

**Step 3-6 should be conducted in the `generator` folder for AMR-to-Text Generation, and the `translator` folder for Syntax-based Machine Translation respectively.** The default settings in this repo should reproduce the results in our paper.

## 3. Vocab & Data Preparation

```
cd generator/tranlator
sh prepare.sh # check it before use
```

## 4. Train

```
cd generator/tranlator
sh train.sh # check it before use
```

## 5. Test

```
cd generator/tranlator
sh work.sh # check it before use

# postprocess
sh test.sh (make sure --output is set)# check it before use
```

## 6. Evaluation

```
for bleu: use sh multi-bleu.perl (-lc)
for chrf++: use python chrF++.py (c6+w2-F2)
for meteor: use meteor-1.5 "java -Xmx2G -jar meteor-1.5.jar test reference -l en"
```

## Citation

If you find the code useful, please cite our paper.
```
@inproceedings{cai-lam-2020-graph,
    title = "Graph Transformer for Graph-to-Sequence Learning",
    author = "Cai, Deng  and Lam, Wai",
    booktitle = "Proceedings of The Thirty-Fourth AAAI Conference on Artificial Intelligence (AAAI)",
    year = "2020",
}
```
## Contact
For any questions, please drop an email to [Deng Cai](https://jcyk.github.io/).

(Pretrained models and our system output are available upon request.)

