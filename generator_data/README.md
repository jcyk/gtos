

## Data Preprocessing for AMR-to-Text Generation

Assuming that you're working on AMR 2.0 ([LDC2017T10](https://catalog.ldc.upenn.edu/LDC2017T10)), unzip the corpus to `data/AMR/LDC2017T10`, and make sure it has the following structure:

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
1. Download Artifacts:

```bash
./scripts/download_artifacts.sh
```

2. Prepare training/dev/test data:

```bash
./scripts/prepare_data.sh -v 2 -p data/AMR/LDC2017T10
```

3. We use [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/index.html) (version **3.9.2**) for tokenizing. First, start a CoreNLP server by `sh run_standford_corenlp_server.sh` Then, annotate AMR sentences:

```bash
sh run_standford_corenlp_server.sh
./scripts/annotate_features.sh data/AMR/amr_2.0
```

4. Data Preprocessing

```bash
./scripts/preprocess_2.0.sh
```
***(Acknowledgements)*** A large body of the code for AMR preprocessing is from [sheng-z/stog](https://github.com/sheng-z/stog).
