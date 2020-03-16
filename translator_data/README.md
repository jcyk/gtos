## Data Preprocessing for Syntax-based Machine Translation
1. You can get the preprocessed data from this [Google Drive](https://drive.google.com/drive/folders/0BxGk3yrG1HHVMy1aYTNld3BIN2s) (thanks to the original authors).
2. Then you should change the data format to the same as `example.txt` by using `get_data.sh (check it before use)`
*(the current script assuming the download path is `gtos/translator_data/data` and the directores to save are `gtos/translator_data/cs` and `gtos/translator_data/de`)*

The above steps will produce data in the following structure.
```
cs
├── dev.txt
├── newstest2015-encs-ref.tok.cs
├── newstest2016-encs-ref.tok.cs
├── test.txt
└── train.txt
de
├── dev.txt
├── newstest2015-ende-ref.tok.de
├── newstest2016-ende-ref.tok.de
├── test.txt
└── train.txt
```
where newstest2015* and newstest2016* store translation references for dev.txt and test.txt respectively (for evaluation).
