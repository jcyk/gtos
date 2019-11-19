#!/usr/bin/env python
# coding: utf-8
from collections import Counter
from dependencyGraph import dependencyGraph

class IO:

    def __init__(self):
        pass

    @staticmethod
    def read(file_path):
        line_id = -1
        for line in open(file_path, encoding='utf-8').readlines():
            line_id += 1
            tp = line_id % 4
            info = line.rstrip('\n').split(' ')
            if tp == 0:
                dep = info
            elif tp == 1:
                head = [int(x) for x in info]
            elif tp == 2:
                tok = info
            else:
                tgt = info
                assert len(dep) == len(head) == len(tok)
                yield dependencyGraph(dep, head, tok, tgt)

    @staticmethod
    def read1(file_path):
        line_id = -1
        for line in open(file_path, encoding='utf-8').readlines():
            line_id += 1
            tp = line_id % 4
            info = line.rstrip('\n').split(' ')
            if tp == 0:
                dep = info
            elif tp == 1:
                head = [int(x) for x in info]
            elif tp == 2:
                tok = info
            else:
                tgt = info
                assert len(dep) == len(head) == len(tok)
                yield dep, head, tok, tgt

class LexicalMap(object):

    # build our lexical mapping (from concept to token/lemma), useful for copy mechanism.
    def __init__(self):
        pass

    #cp_seq, token2idx, idx2token = lex_map.get(concept, vocabs['predictable_token'])
    @staticmethod
    def get(concept, vocab=None):
        cp_seq = []
        for conc in concept:
            cp_seq.append(conc)

        if vocab is None:
            return cp_seq

        new_tokens = set(cp for cp in cp_seq if vocab.token2idx(cp) == vocab.unk_idx)
        token2idx, idx2token = dict(), dict()
        nxt = vocab.size
        for x in new_tokens:
            token2idx[x] = nxt
            idx2token[nxt] = x
            nxt += 1
        return cp_seq, token2idx, idx2token


def read_file(filename):
    # read prepared file
    data = []
    for graph in IO.read(filename):
        data.append(graph)
    print ('read from %s, %d instances'%(filename, len(data)))
    return data

def make_vocab(cnt, char_level=False):
    if not char_level:
        return cnt
    char_cnt = Counter()
    for x, y in cnt.most_common():
        for ch in list(x):
            char_cnt[ch] += y
    return cnt, char_cnt


def write_vocab(vocab, path):
    with open(path, 'w') as fo:
        for x, y in vocab.most_common():
            fo.write('%s\t%d\n'%(x,y))

import argparse
def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str)
    parser.add_argument('--nprocessors', type=int, default=4)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_config()
    lexical_map = LexicalMap()

    token_vocab, conc_vocab, predictable_token_vocab, rel_vocab = Counter(), Counter(), Counter(), Counter()
    for dep, head, concept, token in IO.read1(args.train_data):
        token_vocab.update(token)
        conc_vocab.update(concept)
        lexical_concepts = set(lexical_map.get(concept))
        predictable = [ c for c in token if c not in lexical_concepts]
        predictable_token_vocab.update(predictable)
        rel_vocab.update(dep)
        rel_vocab.update([r+'_r_' for r in dep])

    # make vocabularies
    token_vocab, token_char_vocab = make_vocab(token_vocab, char_level=True)
    conc_vocab, conc_char_vocab = make_vocab(conc_vocab, char_level=True)

    predictable_token_vocab = make_vocab(predictable_token_vocab)

    print ('make vocabularies')
    write_vocab(token_vocab, 'token_vocab')
    write_vocab(token_char_vocab, 'token_char_vocab')
    write_vocab(predictable_token_vocab, 'predictable_token_vocab')
    write_vocab(conc_vocab, 'concept_vocab')
    write_vocab(conc_char_vocab, 'concept_char_vocab')
    write_vocab(rel_vocab, 'relation_vocab')
