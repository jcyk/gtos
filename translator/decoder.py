import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Parameter
from utils import compute_f_by_tensor
from transformer import MultiheadAttention, Transformer

from utils import label_smoothed_nll_loss

class TokenGenerator(nn.Module):
    def __init__(self, vocabs, embed_dim, token_size, dropout):
        super(TokenGenerator, self).__init__()
        self.alignment_layer = MultiheadAttention(embed_dim, 1, dropout, weights_dropout=False)
        self.alignment_layer_norm = nn.LayerNorm(embed_dim)
        self.transfer = nn.Linear(embed_dim, token_size)
        self.generator = nn.Linear(token_size, vocabs['predictable_token'].size)
        self.diverter = nn.Linear(token_size, 2)
        self.vocabs = vocabs
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.transfer.weight, std=0.02)
        nn.init.normal_(self.diverter.weight, std=0.02)
        nn.init.normal_(self.generator.weight, std=0.02) 
        nn.init.constant_(self.diverter.bias, 0.)
        nn.init.constant_(self.transfer.bias, 0.)
        nn.init.constant_(self.generator.bias, 0.)

    def forward(self, outs, graph_state, graph_padding_mask, copy_seq,
                target=None, work=False):
        x, alignment_weight = self.alignment_layer(outs, graph_state, graph_state,
                                                    key_padding_mask=graph_padding_mask,
                                                    need_weights=True)
        x = F.dropout(x, p=self.dropout, training=self.training)
        outs = self.alignment_layer_norm(outs + x)

        seq_len, bsz, _ = outs.size()
        outs_token = torch.tanh(self.transfer(outs))
        outs_token = F.dropout(outs_token, p=self.dropout, training=self.training)

        gen_gate, copy_gate = F.softmax(self.diverter(outs_token), -1).chunk(2, dim=-1)
        
        probs = gen_gate * F.softmax(self.generator(outs_token), -1)

        tot_ext = 1 + copy_seq.max().item()
        vocab_size = probs.size(-1)

        if tot_ext - vocab_size > 0:
            ext_probs = probs.new_zeros((1, 1, tot_ext - vocab_size)).expand(seq_len, bsz, -1)
            probs = torch.cat([probs, ext_probs], -1)

        index = copy_seq.transpose(0, 1).contiguous().view(1, bsz, -1).expand(seq_len, -1, -1)
        
        copy_probs = (copy_gate * alignment_weight).view(seq_len, bsz, -1)
        probs = probs.scatter_add_(-1, index, copy_probs)
        ll = torch.log(probs + 1e-12)
        
        if work:
            return ll

        token_loss = -ll.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
        token_mask = torch.eq(target, self.vocabs['predictable_token'].padding_idx)
        token_loss = token_loss.masked_fill_(token_mask, 0.).sum(0)
        return token_loss

class DecodeLayer(nn.Module):

    def __init__(self, vocabs, inference_layers, embed_dim, ff_embed_dim, num_heads, token_size, rel_size, dropout):
        super(DecodeLayer, self).__init__()
        self.inference_core = Transformer(inference_layers, embed_dim, ff_embed_dim, num_heads, dropout, with_external=True)
        self.token_generator = TokenGenerator(vocabs, embed_dim, token_size, dropout)
        self.dropout = dropout
        self.vocabs = vocabs

    def forward(self, probe, graph_state, snt_state,
                graph_padding_mask, snt_padding_mask, attn_mask,
                copy_seq, target=None, work=False):
        # probe: tgt_len x bsz x embed_dim
        # snt_state, graph_state: seq_len x bsz x embed_dim

        outs = F.dropout(probe, p=self.dropout, training=self.training)
        outs = self.inference_core(outs, kv=snt_state,
                    self_padding_mask=snt_padding_mask, self_attn_mask=attn_mask,
                    external_memories=graph_state, external_padding_mask=graph_padding_mask)

        if work:
            concept_ll = self.token_generator(outs, graph_state, graph_padding_mask, copy_seq, work=True)
            return concept_ll

        token_loss = self.token_generator(outs, graph_state, graph_padding_mask, copy_seq, target=target, work=False)
        token_tot = snt_padding_mask.size(0) - snt_padding_mask.float().sum(0)
        token_loss = token_loss / token_tot
        return token_loss.mean()
