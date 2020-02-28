""" Global attention modules (Luong / Bahdanau) """
import torch
import torch.nn.functional as F

from stog.modules.attention import MLPAttention
from stog.modules.attention import BiaffineAttention


# This class is mainly used by decoder.py for RNNs but also
# by the CNN / transformer decoder when copy attention is used
# CNN has its own attention mechanism ConvMultiStepAttention
# Transformer has its own MultiHeadedAttention


class GlobalAttention(torch.nn.Module):
    """
    Global attention takes a matrix and a query vector. It
    then computes a parameterized convex combination of the matrix
    based on the input query.
    Constructs a unit mapping a query `q` of size `dim`
    and a source matrix `H` of size `n x dim`, to an output
    of size `dim`.
    .. mermaid::
       graph BT
          A[Query]
          subgraph RNN
            C[H 1]
            D[H 2]
            E[H N]
          end
          F[Attn]
          G[Output]
          A --> F
          C --> F
          D --> F
          E --> F
          C -.-> G
          D -.-> G
          E -.-> G
          F --> G
    All models compute the output as
    :math:`c = sum_{j=1}^{SeqLength} a_j H_j` where
    :math:`a_j` is the softmax of a score function.
    Then then apply a projection layer to [q, c].
    However they
    differ on how they compute the attention score.
    * Luong Attention (dot, general):
       * dot: :math:`score(H_j,q) = H_j^T q`
       * general: :math:`score(H_j, q) = H_j^T W_a q`
    * Bahdanau Attention (mlp):
       * :math:`score(H_j, q) = v_a^T tanh(W_a q + U_a h_j)`
    Args:
       dim (int): dimensionality of query and key
       coverage (bool): use coverage term
       attn_type (str): type of attention to use, options [dot,general,mlp]
    """

    def __init__(self, decoder_hidden_size, encoder_hidden_size, attention):
        super(GlobalAttention, self).__init__()
        self.decoder_hidden_size = decoder_hidden_size
        self.encoder_hidden_size = encoder_hidden_size
        self.attention = attention
        self.output_layer = torch.nn.Linear(
            decoder_hidden_size + encoder_hidden_size,
            decoder_hidden_size,
            bias=isinstance(attention, MLPAttention)
        )

    def forward(self, source, memory_bank, mask=None, coverage=None):
        """
        Args:
          source (`FloatTensor`): query vectors `[batch x tgt_len x dim]`
          memory_bank (`FloatTensor`): source vectors `[batch x src_len x dim]`
          mask (`LongTensor`): the source context mask `[batch, length]`
        Returns:
          (`FloatTensor`, `FloatTensor`):
          * Computed vector `[tgt_len x batch x dim]`
          * Attention distribtutions for each query
             `[tgt_len x batch x src_len]`
        """

        batch_, target_l, dim_ = source.size()

        one_step = False
        # one step input
        if source.dim() == 2:
            one_step = True
            source = source.unsqueeze(1)

        if isinstance(self.attention, MLPAttention) and coverage is not None:
            align = self.attention(source, memory_bank, coverage)
        elif isinstance(self.attention, BiaffineAttention):
            align = self.attention(source, memory_bank).squeeze(1)
        else:
            align = self.attention(source, memory_bank)

        if mask is not None:
            mask = mask.byte().unsqueeze(1)  # Make it broadcastable.
            align.masked_fill_(1 - mask, -float('inf'))

        align_vectors = F.softmax(align, 2)

        # each context vector c_t is the weighted average
        # over all the source hidden states
        c = torch.bmm(align_vectors, memory_bank)

        # concatenate
        concat_c = torch.cat([c, source], 2).view(batch_*target_l, -1)
        attn_h = self.output_layer(concat_c).view(batch_, target_l, -1)

        attn_h = torch.tanh(attn_h)

        if coverage is not None:
            coverage = coverage + align_vectors

        if one_step:
            attn_h = attn_h.squeeze(1)
            align_vectors = align_vectors.squeeze(1)

        return attn_h, align_vectors, coverage
