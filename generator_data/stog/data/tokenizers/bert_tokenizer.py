from overrides import overrides

import numpy as np
from pytorch_pretrained_bert.tokenization import BertTokenizer

from stog.data.vocabulary import DEFAULT_PADDING_TOKEN, DEFAULT_OOV_TOKEN


class AMRBertTokenizer(BertTokenizer):

    def __init__(self, *args, **kwargs):
        super(AMRBertTokenizer, self).__init__(*args, **kwargs)

    @overrides
    def tokenize(self, tokens, split=False):
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        if not split:
            split_tokens = [t if t in self.vocab else '[UNK]' for t in tokens]
            gather_indexes = None
        else:
            split_tokens, _gather_indexes = [], []
            for token in tokens:
                indexes = []
                for i, sub_token in enumerate(self.wordpiece_tokenizer.tokenize(token)):
                    indexes.append(len(split_tokens))
                    split_tokens.append(sub_token)
                _gather_indexes.append(indexes)

            _gather_indexes = _gather_indexes[1:-1]
            max_index_list_len = max(len(indexes) for indexes in _gather_indexes)
            gather_indexes = np.zeros((len(_gather_indexes), max_index_list_len))
            for i, indexes in enumerate(_gather_indexes):
                for j, index in enumerate(indexes):
                    gather_indexes[i, j] = index

        token_ids = np.array(self.convert_tokens_to_ids(split_tokens))
        return token_ids, gather_indexes
