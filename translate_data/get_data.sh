pip install subword-nmt
subword-nmt apply-bpe -c en2cs/cs.bpe.8000 < en2cs/news-commentary-v11.cs-en.clean.tok.cs > en2cs/train.ref.bpe
subword-nmt apply-bpe -c en2cs/cs.bpe.8000 < en2cs/newstest2015-encs-ref.tok.cs > en2cs/dev.ref.bpe
subword-nmt apply-bpe -c en2cs/cs.bpe.8000 < en2cs/newstest2016-encs-ref.tok.cs > en2cs/test.ref.bpe
subword-nmt apply-bpe -c en2de/de.bpe.8000 < en2de/news-commentary-v11.de-en.clean.tok.de > en2de/train.ref.bpe
subword-nmt apply-bpe -c en2de/de.bpe.8000 < en2de/newstest2015-ende-ref.tok.de > en2de/dev.ref.bpe
subword-nmt apply-bpe -c en2de/de.bpe.8000 < en2de/newstest2016-ende-ref.tok.de > en2de/test.ref.bpe

python merge.py en2cs/news-commentary-v11.cs-en.clean.deprels.en en2cs/news-commentary-v11.cs-en.clean.heads.en en2cs/news-commentary-v11.cs-en.clean.tok.en en2cs/train.ref.bpe cs/train.txt
python merge.py en2cs/newstest2015-encs-src.deprels.en en2cs/newstest2015-encs-src.heads.en en2cs/newstest2015-encs-src.tok.en en2cs/dev.ref.bpe cs/dev.txt
python merge.py en2cs/newstest2016-encs-src.deprels.en en2cs/newstest2016-encs-src.heads.en en2cs/newstest2016-encs-src.tok.en en2cs/test.ref.bpe cs/test.txt


python merge.py en2de/news-commentary-v11.de-en.clean.deprels.en en2de/news-commentary-v11.de-en.clean.heads.en en2de/news-commentary-v11.de-en.clean.tok.en en2de/train.ref.bpe de/train.txt
python merge.py en2de/newstest2015-ende-src.deprels.en en2de/newstest2015-ende-src.heads.en en2de/newstest2015-ende-src.tok.en en2de/dev.ref.bpe de/dev.txt
python merge.py en2de/newstest2016-ende-src.deprels.en en2de/newstest2016-ende-src.heads.en en2de/newstest2016-ende-src.tok.en en2de/test.ref.bpe de/test.txt