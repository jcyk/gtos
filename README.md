# Graph Transformer

Code for our AAAI2020 paper,

Graph Transformer for Graph-to-Sequence Learning.

Deng Cai and Wai Lam.

## 0. Warning
I haven't have enough time to clean the code and give detailed instructions (will do in the future!).
Therefore, you may need some *adventurous spirits* to use it. However, I am happy to answer questions if you have any.

## 1. Environment Setup

The code runs with Python 3.6.
All dependencies are listed in [requirements.txt](requirements.txt).

`pip install -r requirements.txt`

## 2. Data Preparation
### 2.1 AMR-to-Text
Download Artifacts:
```bash
./scripts/download_artifacts.sh
```

Assuming that you're working on AMR 2.0 ([LDC2017T10](https://catalog.ldc.upenn.edu/LDC2017T10)),
unzip the corpus to `data/AMR/LDC2017T10`, and make sure it has the following structure:

```bash$ tree data/AMR/LDC2017T10 -L 2
data/AMR/LDC2017T10
├── data
│   ├── alignments
│   ├── amrs
│   └── frames
├── docs
│   ├── AMR-alignment-format.txt
│   ├── amr-guidelines-v1.2.pdf
│   ├── file.tbl
│   ├── frameset.dtd
│   ├── PropBank-unification-notes.txt
│   └── README.txt
└── index.html
```

Prepare training/dev/test data:
```bash
./scripts/prepare_data.sh -v 2 -p data/AMR/LDC2017T10
```

We use [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/index.html) (version **3.9.2**) for tokenizing.

First, start a CoreNLP server.
`sh run_standford_corenlp_server.sh`

Then, annotate AMR sentences:
```bash
sh run_standford_corenlp_server.sh
./scripts/annotate_features.sh data/AMR/amr_2.0
```

Data Preprocessing
```bash
./scripts/preprocess_2.0.sh
```
***(Acknowledgements)*** A large body of the code for AMR preprocessing is from [sheng-z/stog](https://github.com/sheng-z/stog).

### 2.2 Syntactic Machine Translation
You can get the preprocessed data from this [Google Dirve](https://drive.google.com/drive/folders/0BxGk3yrG1HHVMy1aYTNld3BIN2s). Thanks to original authors.
Then you should change the data format to the same as `data/example.txt`

## From now on, for AMR-to-Text, you should go to the `generator` folder. For Syntactic Machine Translation, you should go to the `translator` folder.
## 3. Create Vocab & Data Format

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
