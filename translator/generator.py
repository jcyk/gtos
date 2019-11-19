import torch
from torch import nn
import torch.nn.functional as F
import math

from encoder import TokenEncoder, RelationEncoder
from decoder import DecodeLayer
from transformer import Transformer, SinusoidalPositionalEmbedding, SelfAttentionMask
from graph_transformer import GraphTransformer
from data import ListsToTensor, ListsofStringToTensor, STR
from search import Hypothesis, Beam, search_by_batch
from utils import move_to_device

class Generator(nn.Module):
    def __init__(self, vocabs, 
                word_char_dim, word_dim,
                concept_char_dim, concept_dim,
                cnn_filters, char2word_dim, char2concept_dim,
                rel_dim, rnn_hidden_size, rnn_num_layers,
                embed_dim, ff_embed_dim, num_heads, dropout,
                snt_layers, graph_layers, inference_layers,
                pretrained_file, device):
        super(Generator, self).__init__()
        self.vocabs = vocabs
        self.concept_encoder = TokenEncoder(vocabs['concept'], vocabs['concept_char'],
                                          concept_char_dim, concept_dim, embed_dim,
                                          cnn_filters, char2concept_dim, dropout, pretrained_file)
        self.relation_encoder = RelationEncoder(vocabs['relation'], rel_dim, embed_dim, rnn_hidden_size, rnn_num_layers, dropout)
        self.token_encoder = TokenEncoder(vocabs['token'], vocabs['token_char'],
                        word_char_dim, word_dim, embed_dim,
                        cnn_filters, char2word_dim, dropout, pretrained_file)

        self.graph_encoder = GraphTransformer(graph_layers, embed_dim, ff_embed_dim, num_heads, dropout)
        self.snt_encoder = Transformer(snt_layers, embed_dim, ff_embed_dim, num_heads, dropout, with_external=True)
        
        self.embed_dim = embed_dim
        self.embed_scale = math.sqrt(embed_dim)
        self.token_position = SinusoidalPositionalEmbedding(embed_dim, device)
        self.concept_depth = nn.Embedding(256, embed_dim)
        self.token_embed_layer_norm = nn.LayerNorm(embed_dim)
        self.concept_embed_layer_norm = nn.LayerNorm(embed_dim)
        self.self_attn_mask = SelfAttentionMask(device)
        self.decoder = DecodeLayer(vocabs, inference_layers, embed_dim, ff_embed_dim, num_heads, concept_dim, rel_dim, dropout)
        self.dropout = dropout
        self.probe_generator = nn.Linear(embed_dim, embed_dim)
        self.device = device
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.probe_generator.weight, std=0.02)
        nn.init.constant_(self.probe_generator.bias, 0.)
        nn.init.constant_(self.concept_depth.weight, 0.)
    
    def encoder_attn(self, inp):
        with torch.no_grad():
            concept_repr = self.embed_scale * self.concept_encoder(inp['concept'], inp['concept_char']) + self.concept_depth(inp['concept_depth'])
            concept_repr = self.concept_embed_layer_norm(concept_repr)
            concept_mask = torch.eq(inp['concept'], self.vocabs['concept'].padding_idx)

            relation = self.relation_encoder(inp['relation_bank'], inp['relation_length'])
            relation = relation.index_select(0, inp['relation'].contiguous().view(-1)).contiguous().view(*inp['relation'].size(), -1)

            attn = self.graph_encoder.get_attn_weights(concept_repr, relation, self_padding_mask=concept_mask)
            # nlayers x tgt_len x src_len x  bsz x num_heads
        return attn

    def encode_step(self, inp):
        concept_repr = self.embed_scale * self.concept_encoder(inp['concept'], inp['concept_char']) + self.concept_depth(inp['concept_depth'])
        concept_repr = self.concept_embed_layer_norm(concept_repr)
        concept_mask = torch.eq(inp['concept'], self.vocabs['concept'].padding_idx)

        relation = self.relation_encoder(inp['relation_bank'], inp['relation_length'])
        relation = relation.index_select(0, inp['relation'].contiguous().view(-1)).contiguous().view(*inp['relation'].size(), -1)

        concept_repr = self.graph_encoder(concept_repr, relation, self_padding_mask=concept_mask)

        probe = torch.tanh(self.probe_generator(concept_repr[:1]))
        concept_repr = concept_repr[1:]
        concept_mask = concept_mask[1:]
        return concept_repr, concept_mask, probe

    def work(self, data, beam_size, max_time_step, min_time_step=1):
        with torch.no_grad():
            concept_repr, concept_mask, probe = self.encode_step(data)

            mem_dict = {'graph_state':concept_repr,
                        'graph_padding_mask':concept_mask,
                        'probe':probe,
                        'local_idx2token':data['local_idx2token'],
                        'cp_seq':data['cp_seq']}
            init_state_dict = {}
            init_hyp = Hypothesis(init_state_dict, [STR], 0.)
            bsz = concept_repr.size(1)
            beams = [ Beam(beam_size, min_time_step, max_time_step, [init_hyp], self.device) for i in range(bsz)]
            search_by_batch(self, beams, mem_dict)
        return beams


    def prepare_incremental_input(self, step_seq):
        token = ListsToTensor(step_seq, self.vocabs['token'])
        token_char = ListsofStringToTensor(step_seq, self.vocabs['token_char'])
        token, token_char = move_to_device(token, self.device), move_to_device(token_char, self.device)
        return token, token_char

    def decode_step(self, inp, state_dict, mem_dict, offset, topk): 
        step_token, step_token_char = inp
        graph_repr = mem_dict['graph_state']
        graph_padding_mask = mem_dict['graph_padding_mask']
        probe = mem_dict['probe']
        copy_seq = mem_dict['cp_seq']
        local_vocabs = mem_dict['local_idx2token']
        _, bsz, _ = graph_repr.size()

        new_state_dict = {}

        token_repr = self.embed_scale * self.token_encoder(step_token, step_token_char) + self.token_position(step_token, offset)
        token_repr = self.token_embed_layer_norm(token_repr)
        for idx, layer in enumerate(self.snt_encoder.layers):
            name_i = 'token_repr_%d'%idx
            if name_i in state_dict:
                prev_token_repr = state_dict[name_i]
                new_token_repr = torch.cat([prev_token_repr, token_repr], 0)
            else:
                new_token_repr = token_repr

            new_state_dict[name_i] = new_token_repr
            token_repr, _, _ = layer(token_repr, kv=new_token_repr, external_memories=graph_repr, external_padding_mask=graph_padding_mask)
        name = 'token_state'
        if name in state_dict:
            prev_token_state = state_dict[name]
            new_token_state = torch.cat([prev_token_state, token_repr], 0)
        else:
            new_token_state = token_repr
        new_state_dict[name] = new_token_state
        LL = self.decoder(probe, graph_repr, new_token_state, graph_padding_mask, None, None, copy_seq, work=True)


        def idx2token(idx, local_vocab):
            if idx in local_vocab:
                return local_vocab[idx]
            return self.vocabs['predictable_token'].idx2token(idx)

        topk_scores, topk_token = torch.topk(LL.squeeze(0), topk, 1) # bsz x k

        results = []
        for s, t, local_vocab in zip(topk_scores.tolist(), topk_token.tolist(), local_vocabs):
            res = []
            for score, token in zip(s, t):
                res.append((idx2token(token, local_vocab), score))
            results.append(res)

        return new_state_dict, results

    def forward(self, data):
        concept_repr, concept_mask, probe = self.encode_step(data)
        token_repr = self.embed_scale * self.token_encoder(data['token_in'], data['token_char_in']) + self.token_position(data['token_in'])
        token_repr = self.token_embed_layer_norm(token_repr)
        token_repr = F.dropout(token_repr, p=self.dropout, training=self.training)
        token_mask = torch.eq(data['token_in'], self.vocabs['token'].padding_idx)
        attn_mask = self.self_attn_mask(data['token_in'].size(0))
        token_repr = self.snt_encoder(token_repr,
                                  self_padding_mask=token_mask, self_attn_mask=attn_mask,
                                  external_memories=concept_repr, external_padding_mask=concept_mask)
        
        probe = probe.expand_as(token_repr) # tgt_len x bsz x embed_dim
        return self.decoder(probe, concept_repr, token_repr, concept_mask, token_mask, attn_mask, \
         data['cp_seq'], target=data['token_out'])
