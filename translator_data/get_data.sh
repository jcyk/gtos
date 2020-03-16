pip install subword-nmt
subword-nmt apply-bpe -c data/en2cs/cs.bpe.8000 < data/en2cs/news-commentary-v11.cs-en.clean.tok.cs > data/en2cs/train.ref.bpe
subword-nmt apply-bpe -c data/en2cs/cs.bpe.8000 < data/dev/newstest2015-encs-ref.tok.cs > data/en2cs/dev.ref.bpe
subword-nmt apply-bpe -c data/en2cs/cs.bpe.8000 < data/test/newstest2016-encs-ref.tok.cs > data/en2cs/test.ref.bpe
subword-nmt apply-bpe -c data/en2de/de.bpe.8000 < data/en2de/news-commentary-v11.de-en.clean.tok.de > data/en2de/train.ref.bpe
subword-nmt apply-bpe -c data/en2de/de.bpe.8000 < data/dev/newstest2015-ende-ref.tok.de > data/en2de/dev.ref.bpe
subword-nmt apply-bpe -c data/en2de/de.bpe.8000 < data/test/newstest2016-ende-ref.tok.de > data/en2de/test.ref.bpe

python merge.py data/en2cs/news-commentary-v11.cs-en.clean.deprels.en data/en2cs/news-commentary-v11.cs-en.clean.heads.en data/en2cs/news-commentary-v11.cs-en.clean.tok.en data/en2cs/train.ref.bpe data/cs/train.txt
python merge.py data/dev/newstest2015-encs-src.deprels.en data/dev/newstest2015-encs-src.heads.en data/dev/newstest2015-encs-src.tok.en data/en2cs/dev.ref.bpe data/cs/dev.txt
python merge.py data/test/newstest2016-encs-src.deprels.en data/test/newstest2016-encs-src.heads.en data/test/newstest2016-encs-src.tok.en data/en2cs/test.ref.bpe data/cs/test.txt


python merge.py data/en2de/news-commentary-v11.de-en.clean.deprels.en data/en2de/news-commentary-v11.de-en.clean.heads.en data/en2de/news-commentary-v11.de-en.clean.tok.en data/en2de/train.ref.bpe data/de/train.txt
python merge.py data/dev/newstest2015-ende-src.deprels.en data/dev/newstest2015-ende-src.heads.en data/dev/newstest2015-ende-src.tok.en data/en2de/dev.ref.bpe data/de/dev.txt
python merge.py data/test/newstest2016-ende-src.deprels.en data/test/newstest2016-ende-src.heads.en data/test/newstest2016-ende-src.tok.en data/en2de/test.ref.bpe data/de/test.txt

cp data/dev/newstest2015-encs-ref.tok.cs cs/newstest2015-encs-ref.tok.cs
cp data/test/newstest2016-encs-ref.tok.cs cs/newstest2016-encs-ref.tok.cs

cp data/dev/newstest2015-ende-ref.tok.de de/newstest2015-ende-ref.tok.de
cp data/test/newstest2016-ende-ref.tok.de de/newstest2016-ende-ref.tok.de
