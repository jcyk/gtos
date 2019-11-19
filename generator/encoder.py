import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from transformer import Embedding
import re

def AMREmbedding(vocab, embedding_dim, pretrained_file=None, amr=False, dump_file=None):
    if pretrained_file is None:
        return Embedding(vocab.size, embedding_dim, vocab.padding_idx)

    tokens_to_keep = set()
    for idx in range(vocab.size):
        token = vocab.idx2token(idx)
        # TODO: Is there a better way to do this? Currently we have a very specific 'amr' param.
        if amr:
            token = re.sub(r'-\d\d$', '', token)
        tokens_to_keep.add(token)

    embeddings = {}
 
    if dump_file is not None:
        fo = open(dump_file, 'w', encoding='utf8')

    with open(pretrained_file, encoding='utf8') as embeddings_file:
        for line in embeddings_file.readlines():    
            fields = line.rstrip().split(' ')
            if len(fields) - 1 != embedding_dim:
                continue
            token = fields[0]
            if token in tokens_to_keep:
                if dump_file is not None:
                    fo.write(line)
                vector = np.asarray(fields[1:], dtype='float32')
                embeddings[token] = vector

    if dump_file is not None:
        fo.close()

    all_embeddings = np.asarray(list(embeddings.values()))
    print ('pretrained', all_embeddings.shape)
    embeddings_mean = float(np.mean(all_embeddings))
    embeddings_std = float(np.std(all_embeddings))
    # Now we initialize the weight matrix for an embedding layer, starting with random vectors,
    # then filling in the word vectors we just read.
    embedding_matrix = torch.FloatTensor(vocab.size, embedding_dim).normal_(embeddings_mean,
                                                                            embeddings_std)

    for i in range(vocab.size):
        token = vocab.idx2token(i)

        # If we don't have a pre-trained vector for this word, we'll just leave this row alone,
        # so the word has a random initialization.
        if token in embeddings:
            embedding_matrix[i] = torch.FloatTensor(embeddings[token])
        else:
            if amr:
                normalized_token = re.sub(r'-\d\d$', '', token)
                if normalized_token in embeddings:
                    embedding_matrix[i] = torch.FloatTensor(embeddings[normalized_token])
    embedding_matrix[vocab.padding_idx].fill_(0.)

    return nn.Embedding.from_pretrained(embedding_matrix, freeze=False)

class RelationEncoder(nn.Module):
    def __init__(self, vocab, rel_dim, embed_dim, hidden_size, num_layers, dropout, bidirectional=True):
        super(RelationEncoder, self).__init__()
        self.vocab  = vocab
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.rel_embed = AMREmbedding(vocab, rel_dim)
        self.rnn = nn.GRU(
            input_size=rel_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=self.dropout if num_layers > 1 else 0.,
            bidirectional=bidirectional
        )
        tot_dim = 2 * hidden_size if bidirectional else hidden_size
        self.out_proj = nn.Linear(tot_dim, embed_dim)

    def reset_parameters(self):
        nn.init.normal_(self.out_proj.weight, std=0.02)
        nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, src_tokens, src_lengths):
        seq_len, bsz = src_tokens.size()
        ###
        sorted_src_lengths, indices = torch.sort(src_lengths, descending=True)
        sorted_src_tokens = src_tokens.index_select(1, indices)
        ###
        x = self.rel_embed(sorted_src_tokens)
        x = F.dropout(x, p=self.dropout, training=self.training)

        packed_x = nn.utils.rnn.pack_padded_sequence(x, sorted_src_lengths.data.tolist())
 
        if self.bidirectional:
            state_size = 2 * self.num_layers, bsz, self.hidden_size
        else:
            state_size = self.num_layers, bsz, self.hidden_size 
        h0 = x.data.new(*state_size).zero_()
        _, final_h = self.rnn(packed_x, h0)

        if self.bidirectional:
            def combine_bidir(outs):
                return outs.view(self.num_layers, 2, bsz, -1).transpose(1, 2).contiguous().view(self.num_layers, bsz, -1)
            final_h = combine_bidir(final_h)

        ###
        _, positions = torch.sort(indices)
        final_h = final_h.index_select(1, positions) # num_layers x bsz x hidden_size

        output = self.out_proj(final_h[-1]) 

        return output



class TokenEncoder(nn.Module):
    def __init__(self, token_vocab, char_vocab, char_dim, token_dim, embed_dim, filters, char2token_dim, dropout, pretrained_file=None):
        super(TokenEncoder, self).__init__()
        self.char_embed = AMREmbedding(char_vocab, char_dim)
        self.token_embed = AMREmbedding(token_vocab, token_dim, pretrained_file)
        self.char2token = CNNEncoder(filters, char_dim, char2token_dim)
        tot_dim = char2token_dim + token_dim
        self.out_proj = nn.Linear(tot_dim, embed_dim)
        self.char_dim = char_dim
        self.token_dim = token_dim
        self.dropout = dropout

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.out_proj.weight, std=0.02)
        nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, token_input, char_input):
        seq_len, bsz, _ = char_input.size()
        char_repr = self.char_embed(char_input.view(seq_len * bsz, -1))
        char_repr = self.char2token(char_repr).view(seq_len, bsz, -1)
        token_repr = self.token_embed(token_input)

        token = F.dropout(torch.cat([char_repr,token_repr], -1), p=self.dropout, training=self.training)
        token = self.out_proj(token)
        return token

class CNNEncoder(nn.Module):
    def __init__(self, filters, input_dim, output_dim, highway_layers=1):
        super(CNNEncoder, self).__init__()
        self.convolutions = nn.ModuleList()
        for width, out_c in filters:
            self.convolutions.append(nn.Conv1d(input_dim, out_c, kernel_size=width))
        final_dim = sum(f[1] for f in filters)
        self.highway = Highway(final_dim, highway_layers)
        self.out_proj = nn.Linear(final_dim, output_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.out_proj.weight, std=0.02)
        nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, input):
        # input: batch_size x seq_len x input_dim
        x  = input.transpose(1, 2)
        conv_result = []
        for i, conv in enumerate(self.convolutions):
            y = conv(x)
            y, _ = torch.max(y, -1)
            y = F.relu(y)
            conv_result.append(y)

        conv_result = torch.cat(conv_result, dim=-1)
        conv_result = self.highway(conv_result)
        return self.out_proj(conv_result) #  batch_size x output_dim

class Highway(nn.Module):
    def __init__(self, input_dim, layers):
        super(Highway, self).__init__()
        self.input_dim = input_dim
        self.layers = nn.ModuleList([nn.Linear(input_dim, input_dim * 2)
                                     for _ in range(layers)])
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            nn.init.normal_(layer.weight, std=0.02)
            nn.init.constant_(layer.bias[self.input_dim:], 1)
            nn.init.constant_(layer.bias[:self.input_dim], 0)

    def forward(self, x):
        for layer in self.layers:
            new_x = layer(x)
            new_x, gate = new_x.chunk(2, dim=-1)
            new_x = F.relu(new_x)
            gate = torch.sigmoid(gate)
            x = gate * x + (1 - gate) * new_x
        return x
