import torch
import sacrebleu
import json

from data import Vocab,  DataLoader, STR, END, CLS, rCLS, SEL, TL
from generator import Generator
from extract import LexicalMap, read_file
from utils import move_to_cuda
from postprocess import PostProcess
import argparse, os

def parse_config():
    parser = argparse.ArgumentParser()

    parser.add_argument('--load_path', type=str)
    parser.add_argument('--test_data', type=str)
    parser.add_argument('--test_batch_size', type=int)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--output_suffix', type=str)

    return parser.parse_args()

def record_batch(model, batch, data):
    batch = move_to_cuda(batch, model.device)
    attn = model.encoder_attn(batch)
    #nlayers x tgt_len x src_len x  bsz x num_heads

    for i, x in enumerate(data):
        L = len(x['concept'])+1
        x['attn'] = attn[:,:L,:L,i,:].cpu()
    return data

if __name__ == "__main__":

    args = parse_config()

    test_models = []
    if os.path.isdir(args.load_path):
        for file in os.listdir(args.load_path):
            fname = os.path.join(args.load_path, file)
            if os.path.isfile(fname):
                test_models.append(fname)
        model_args = torch.load(fname)['args']  
    else:
        test_models.append(args.load_path)
        model_args = torch.load(args.load_path)['args']
    vocabs = dict()
    vocabs['concept'] = Vocab(model_args.concept_vocab, 5, [CLS])
    vocabs['token'] = Vocab(model_args.token_vocab, 5, [STR, END])
    vocabs['predictable_token'] = Vocab(model_args.predictable_token_vocab, 5, [END])
    vocabs['token_char'] = Vocab(model_args.token_char_vocab, 100, [STR, END])
    vocabs['concept_char'] = Vocab(model_args.concept_char_vocab, 100, [STR, END])
    vocabs['relation'] = Vocab(model_args.relation_vocab, 5, [CLS, rCLS, SEL, TL])
    lexical_mapping = LexicalMap()

    if args.device < 0:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda', args.device)

    model = Generator(vocabs,
            model_args.token_char_dim, model_args.token_dim,
            model_args.concept_char_dim, model_args.concept_dim,
            model_args.cnn_filters, model_args.char2word_dim, model_args.char2concept_dim,
            model_args.rel_dim, model_args.rnn_hidden_size, model_args.rnn_num_layers,
            model_args.embed_dim, model_args.ff_embed_dim, model_args.num_heads, model_args.dropout,
            model_args.snt_layers, model_args.graph_layers, model_args.inference_layers,
            model_args.pretrained_file,
            device)

    test_data = DataLoader(vocabs, lexical_mapping, args.test_data, args.test_batch_size, for_train=False)
    test_data.record()
    for test_model in test_models:
        print (test_model)
        model.load_state_dict(torch.load(test_model)['model'])
        model = model.to(device)
        model.eval()
        nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print ('total number of parameters %d'%nparams)
        tot = []
        for batch, data in test_data:
            res = record_batch(model, batch, data)
            tot.extend(res)
        print (len(tot))
        torch.save(tot, test_model+args.output_suffix)
