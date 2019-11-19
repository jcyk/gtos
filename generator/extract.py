#!/usr/bin/env python
# coding: utf-8
from smatch import AMR
from AMRGraph import AMRGraph, number_regexp
from collections import Counter
import json, re
from AMRGraph import  _is_abs_form
from multiprocessing import Pool

class AMRIO:

    def __init__(self):
        pass

    @staticmethod
    def read(file_path):
        with open(file_path, encoding='utf-8') as f:
            for line in f:
                line = line.rstrip()
                if line.startswith('# ::id '):
                    amr_id = line[len('# ::id '):]
                elif line.startswith('# ::snt '):
                    sentence = line[len('# ::snt '):]
                elif line.startswith('# ::tokens '):
                    tokens = json.loads(line[len('# ::tokens '):])
                    tokens = [ to if _is_abs_form(to) else to.lower() for to in tokens]
                elif line.startswith('# ::lemmas '):
                    lemmas = json.loads(line[len('# ::lemmas '):])
                    lemmas = [ le if _is_abs_form(le) else le.lower() for le in lemmas]
                elif line.startswith('# ::pos_tags '):
                    pos_tags = json.loads(line[len('# ::pos_tags '):])
                elif line.startswith('# ::ner_tags '):
                    ner_tags = json.loads(line[len('# ::ner_tags '):])
                elif line.startswith('# ::abstract_map '):
                    abstract_map = json.loads(line[len('# ::abstract_map '):])
                    graph_line = AMR.get_amr_line(f)
                    amr = AMR.parse_AMR_line(graph_line)
                    myamr = AMRGraph(amr)
                    yield tokens, lemmas, abstract_map, myamr

class LexicalMap(object):

    # build our lexical mapping (from concept to token/lemma), useful for copy mechanism.
    def __init__(self):
        pass

    #cp_seq, token2idx, idx2token = lex_map.get(concept, vocabs['predictable_token'])
    def get(self, concept, vocab=None):
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
    # read preprocessed amr file
    token, lemma, abstract, amrs = [], [], [], []
    for _tok, _lem, _abstract, _myamr in AMRIO.read(filename):
        token.append(_tok)
        lemma.append(_lem)
        abstract.append(_abstract)
        amrs.append(_myamr)
    print ('read from %s, %d amrs'%(filename, len(token)))
    return amrs, token, lemma, abstract

def make_vocab(batch_seq, char_level=False):
    cnt = Counter()
    for seq in batch_seq:
        cnt.update(seq)
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
    parser.add_argument('--amr_files', type=str, nargs='+')
    parser.add_argument('--nprocessors', type=int, default=4)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_config()
    amrs, token, lemma, abstract = read_file(args.train_data)
    lexical_map = LexicalMap()

    # collect concepts and relations

    def work(data):
        amr, lem, tok = data
        concept, depth, relation, ok = amr.collect_concepts_and_relations()
        assert ok, "not connected"
        lexical_concepts = set(lexical_map.get(concept))
        predictable = [ c for c in tok if c not in lexical_concepts]
        return concept, depth, relation, predictable
    pool = Pool(args.nprocessors)
    res = pool.map(work, zip(amrs, lemma, token), len(amrs)//args.nprocessors)

    tot_pairs = 0
    multi_path_pairs = 0
    tot_paths = 0
    extreme_long_paths = 0
    avg_path_length = 0.
    conc, rel, predictable_token = [], [], []
    for concept, depth, relation, predictable in res:
        conc.append(concept)
        predictable_token.append(predictable)
        for x in relation:
            for y in relation[x]:
                tot_pairs += 1
                if len(relation[x][y]) > 1:
                    multi_path_pairs +=1
                for path in relation[x][y]:
                    tot_paths += 1
                    path_len = path['length']
                    rel.append(path['edge'])
                    if path_len > 8:
                        extreme_long_paths += 1
                    avg_path_length += path_len
    avg_path_length  = avg_path_length / tot_paths
    print ('tot_paths', tot_paths, 'avg_path_length', avg_path_length)
    print ('extreme_long_paths', extreme_long_paths, \
           'extreme_long_paths_percentage', extreme_long_paths/tot_paths)
    print ('multi_path_percentage', multi_path_pairs, tot_pairs, multi_path_pairs/tot_pairs)
    # make vocabularies
    token_vocab, token_char_vocab = make_vocab(token, char_level=True)
    lemma_vocab, lemma_char_vocab = make_vocab(lemma, char_level=True)
    conc_vocab, conc_char_vocab = make_vocab(conc, char_level=True)

    predictable_token_vocab = make_vocab(predictable_token)
    num_predictable_token = sum(len(x) for x in predictable_token)
    num_token = sum(len(x) for x in token)
    print ('predictable token coverage (1. - copyable token coverage)', num_predictable_token, num_token, num_predictable_token/num_token)
    rel_vocab = make_vocab(rel)

    print ('make vocabularies')
    write_vocab(token_vocab, 'token_vocab')
    write_vocab(token_char_vocab, 'token_char_vocab')
    write_vocab(predictable_token_vocab, 'predictable_token_vocab')
    #write_vocab(lemma_vocab, 'lem_vocab')
    #write_vocab(lemma_char_vocab, 'lem_char_vocab')
    write_vocab(conc_vocab, 'concept_vocab')
    write_vocab(conc_char_vocab, 'concept_char_vocab')
    write_vocab(rel_vocab, 'relation_vocab')

    for file in args.amr_files:
        my_data = []
        amrs, token, lemma, abstract = read_file(file)
        res = pool.map(work, zip(amrs, lemma, token), len(amrs)//args.nprocessors)
        for gr, to, le, ab in zip(res, token, lemma, abstract):
            concept, depth, relation, _ = gr
            item = {
                'concept': concept,
                'depth': depth,
                'relation': relation,
                'token': to,
                'lemma': le,
                'abstract': ab
            }
            my_data.append(item)
        json.dump(my_data, open(file+'.json', 'w', encoding='utf-8'))
