import numpy as np
import torch

from stog.models.model import Model
from stog.utils.nn import masked_log_softmax
from stog.modules.attention import BiaffineAttention
from stog.modules.linear import BiLinear
from stog.metrics import AttachmentScores
from stog.algorithms.maximum_spanning_tree import decode_mst_with_coreference, decode_mst


class DeepBiaffineGraphDecoder(torch.nn.Module):

    def __init__(
            self,
            decode_algorithm,
            head_sentinel,
            edge_node_h_linear,
            edge_node_m_linear,
            edge_label_h_linear,
            edge_label_m_linear,
            encode_dropout,
            biaffine_attention,
            edge_label_bilinear
    ):
        super(DeepBiaffineGraphDecoder, self).__init__()
        self.decode_algorithm = decode_algorithm
        self.head_sentinel = head_sentinel
        self.edge_node_h_linear = edge_node_h_linear
        self.edge_node_m_linear = edge_node_m_linear
        self.edge_label_h_linear = edge_label_h_linear
        self.edge_label_m_linear = edge_label_m_linear
        self.encode_dropout = encode_dropout
        self.biaffine_attention = biaffine_attention
        self.edge_label_bilinear = edge_label_bilinear

        self.metrics = AttachmentScores()

        self.minus_inf = -1e8

    def forward(self, memory_bank, edge_heads, edge_labels, corefs, mask):
        num_nodes = mask.sum().item()

        memory_bank, edge_heads, edge_labels, corefs, mask = self._add_head_sentinel(
            memory_bank, edge_heads, edge_labels, corefs, mask)

        (edge_node_h, edge_node_m), (edge_label_h, edge_label_m) = self.encode(memory_bank)

        edge_node_scores = self._get_edge_node_scores(edge_node_h, edge_node_m, mask)

        edge_node_nll, edge_label_nll = self.get_loss(
            edge_label_h, edge_label_m, edge_node_scores, edge_heads, edge_labels, mask)

        pred_edge_heads, pred_edge_labels = self.decode(
            edge_label_h, edge_label_m, edge_node_scores, corefs, mask)

        self.metrics(
            pred_edge_heads, pred_edge_labels, edge_heads[:, 1:], edge_labels[:, 1:], mask[:, 1:],
            edge_node_nll.item(), edge_label_nll.item()
        )

        return dict(
            edge_heads=pred_edge_heads,
            edge_labels=pred_edge_labels,
            loss=(edge_node_nll + edge_label_nll) / num_nodes,
            total_loss=edge_node_nll + edge_label_nll,
            num_nodes=torch.tensor(float(num_nodes)).type_as(memory_bank)
        )

    def encode(self, memory_bank):
        """
        Map contextual representation into specific space (w/ lower dimensionality).

        :param input: [batch, length, hidden_size]
        :return:
            edge_node: a tuple of (head, modifier) hidden state with size [batch, length, edge_hidden_size]
            edge_label: a tuple of (head, modifier) hidden state with size [batch, length, label_hidden_size]
        """

        # Output: [batch, length, edge_hidden_size]
        edge_node_h = torch.nn.functional.elu(self.edge_node_h_linear(memory_bank))
        edge_node_m = torch.nn.functional.elu(self.edge_node_m_linear(memory_bank))

        # Output: [batch, length, label_hidden_size]
        edge_label_h = torch.nn.functional.elu(self.edge_label_h_linear(memory_bank))
        edge_label_m = torch.nn.functional.elu(self.edge_label_m_linear(memory_bank))

        # Apply dropout to certain node?
        # [batch, length * 2, hidden_size]
        edge_node = torch.cat([edge_node_h, edge_node_m], dim=1)
        edge_label = torch.cat([edge_label_h, edge_label_m], dim=1)
        edge_node = self.encode_dropout(edge_node.transpose(1, 2)).transpose(1, 2)
        edge_label = self.encode_dropout(edge_label.transpose(1, 2)).transpose(1, 2)

        edge_node_h, edge_node_m = edge_node.chunk(2, 1)
        edge_label_h, edge_label_m = edge_label.chunk(2, 1)

        return (edge_node_h, edge_node_m), (edge_label_h, edge_label_m)

    def get_loss(self, edge_label_h, edge_label_m, edge_node_scores, edge_heads, edge_labels, mask):
        """
        :param edge_label_h: [batch, length, hidden_size]
        :param edge_label_m: [batch, length, hidden_size]
        :param edge_node_scores:  [batch, length, length]
        :param edge_heads:  [batch, length]
        :param edge_labels:  [batch, length]
        :param mask: [batch, length]
        :return:  [batch, length - 1]
        """
        batch_size, max_len, _ = edge_node_scores.size()

        edge_node_log_likelihood = masked_log_softmax(
            edge_node_scores, mask.unsqueeze(2) + mask.unsqueeze(1), dim=1)

        edge_label_scores = self._get_edge_label_scores(edge_label_h, edge_label_m, edge_heads)
        edge_label_log_likelihood = torch.nn.functional.log_softmax(edge_label_scores, dim=2)

        # Create indexing matrix for batch: [batch, 1]
        batch_index = torch.arange(0, batch_size).view(batch_size, 1).type_as(edge_heads)
        # Create indexing matrix for modifier: [batch, modifier_length]
        modifier_index = torch.arange(0, max_len).view(1, max_len).expand(batch_size, max_len).type_as(edge_heads)
        # Index the log likelihood of gold edges.
        _edge_node_log_likelihood = edge_node_log_likelihood[
            batch_index, edge_heads.data, modifier_index]
        _edge_label_log_likelihood = edge_label_log_likelihood[
            batch_index, modifier_index, edge_labels.data]

        # Exclude the dummy root.
        # Output [batch, length - 1]
        gold_edge_node_nll = - _edge_node_log_likelihood[:, 1:].sum()
        gold_edge_label_nll = - _edge_label_log_likelihood[:, 1:].sum()

        return gold_edge_node_nll, gold_edge_label_nll

    def decode(self, edge_label_h, edge_label_m, edge_node_scores, corefs, mask):
        if self.decode_algorithm == 'mst':
            return self.mst_decode(edge_label_h, edge_label_m, edge_node_scores, corefs, mask)
        else:
            return self.greedy_decode(edge_label_h, edge_label_m, edge_node_scores, mask)

    def greedy_decode(self, edge_label_h, edge_label_m, edge_node_scores, mask):
        # out_arc shape [batch, length, length]
        edge_node_scores = edge_node_scores.data
        max_len = edge_node_scores.size(1)

        # Set diagonal elements to -inf
        edge_node_scores = edge_node_scores + torch.diag(edge_node_scores.new(max_len).fill_(-np.inf))

        # Set invalid positions to -inf
        minus_mask = (1 - mask.float()) * self.minus_inf
        edge_node_scores = edge_node_scores + minus_mask.unsqueeze(2) + minus_mask.unsqueeze(1)

        # Compute naive predictions.
        # prediction shape = [batch, length]
        _, edge_heads = edge_node_scores.max(dim=1)

        # Based on predicted heads, compute the edge label scores.
        # [batch, length, num_labels]
        edge_label_scores = self._get_edge_label_scores(edge_label_h, edge_label_m, edge_heads)
        _, edge_labels = edge_label_scores.max(dim=2)

        return edge_heads[:, 1:], edge_labels[:, 1:]

    def mst_decode(self, edge_label_h, edge_label_m, edge_node_scores, corefs, mask):
        batch_size, max_length, edge_label_hidden_size = edge_label_h.size()
        lengths = mask.data.sum(dim=1).long().cpu().numpy()

        expanded_shape = [batch_size, max_length, max_length, edge_label_hidden_size]
        edge_label_h = edge_label_h.unsqueeze(2).expand(*expanded_shape).contiguous()
        edge_label_m = edge_label_m.unsqueeze(1).expand(*expanded_shape).contiguous()
        # [batch, max_head_length, max_modifier_length, num_labels]
        edge_label_scores = self.edge_label_bilinear(edge_label_h, edge_label_m)
        edge_label_scores = torch.nn.functional.log_softmax(edge_label_scores, dim=3).permute(0, 3, 1, 2)

        # Set invalid positions to -inf
        minus_mask = (1 - mask.float()) * self.minus_inf
        edge_node_scores = edge_node_scores + minus_mask.unsqueeze(2) + minus_mask.unsqueeze(1)
        # [batch, max_head_length, max_modifier_length]
        edge_node_scores = torch.nn.functional.log_softmax(edge_node_scores, dim=1)

        # [batch, num_labels, max_head_length, max_modifier_length]
        batch_energy = torch.exp(edge_node_scores.unsqueeze(1) + edge_label_scores)

        edge_heads, edge_labels = self._run_mst_decoding(batch_energy, lengths, corefs)
        return edge_heads[:, 1:], edge_labels[:, 1:]

    @staticmethod
    def _run_mst_decoding(batch_energy, lengths, corefs=None):
        edge_heads = []
        edge_labels = []
        for i, (energy, length) in enumerate(zip(batch_energy.detach().cpu(), lengths)):
            # energy: [num_labels, max_head_length, max_modifier_length]
            # scores | label_ids : [max_head_length, max_modifier_length]
            scores, label_ids = energy.max(dim=0)
            # Although we need to include the root node so that the MST includes it,
            # we do not want the dummy root node to be the head of more than one nodes,
            # since there should be only one root in a sentence.
            # Here, we enforce this by setting the scores for all word -> ROOT edges
            # edges to be 0.
            # TODO: set it to -1 seems better?
            scores[0, :] = 0
            # Decode the heads. Because we modify the scores to prevent
            # adding in word -> ROOT edges, we need to find the labels ourselves.
            if corefs is not None:
                coref = corefs[i].detach().cpu().tolist()[:length]
                instance_heads, _ = decode_mst_with_coreference(
                    scores.numpy(), coref, length, has_labels=False)
            else:
                instance_heads, _ = decode_mst(scores.numpy(), length, has_labels=False)

            # Find the labels which correspond to the edges in the max spanning tree.
            instance_head_labels = []
            for child, parent in enumerate(instance_heads):
                instance_head_labels.append(label_ids[parent, child].item())
            # We don't care what the head or tag is for the root token, but by default it's
            # not necessarily the same in the batched vs unbatched case, which is annoying.
            # Here we'll just set them to zero.
            instance_heads[0] = 0
            instance_head_labels[0] = 0
            edge_heads.append(instance_heads)
            edge_labels.append(instance_head_labels)
        return torch.from_numpy(np.stack(edge_heads)), torch.from_numpy(np.stack(edge_labels))

    @classmethod
    def from_params(cls, vocab, params):
        decode_algorithm = params['decode_algorithm']
        input_size = params['input_size']
        edge_node_hidden_size = params['edge_node_hidden_size']
        edge_label_hidden_size = params['edge_label_hidden_size']
        dropout = params['dropout']

        head_sentinel = torch.nn.Parameter(torch.randn([1, 1, input_size]))

        # Transform representations into a space for edge node heads, edge node modifiers
        edge_node_h_linear = torch.nn.Linear(input_size, edge_node_hidden_size)
        edge_node_m_linear = torch.nn.Linear(input_size, edge_node_hidden_size)

        # Transform representations into a space for edge label heads, edge label modifiers
        edge_label_h_linear = torch.nn.Linear(input_size, edge_label_hidden_size)
        edge_label_m_linear = torch.nn.Linear(input_size, edge_label_hidden_size)

        encode_dropout = torch.nn.Dropout2d(p=dropout)

        biaffine_attention = BiaffineAttention(edge_node_hidden_size, edge_node_hidden_size)

        num_labels = vocab.get_vocab_size("head_tags")
        edge_label_bilinear = BiLinear(edge_label_hidden_size, edge_label_hidden_size, num_labels)

        return cls(
            decode_algorithm=decode_algorithm,
            head_sentinel=head_sentinel,
            edge_node_h_linear=edge_node_h_linear,
            edge_node_m_linear=edge_node_m_linear,
            edge_label_h_linear=edge_label_h_linear,
            edge_label_m_linear=edge_label_m_linear,
            encode_dropout=encode_dropout,
            biaffine_attention=biaffine_attention,
            edge_label_bilinear=edge_label_bilinear
        )

    def _add_head_sentinel(self, memory_bank, edge_heads, edge_labels, corefs, mask):
        """
        Add a dummy ROOT at the beginning of each node sequence.
        :param memory_bank: [batch, length, hidden_size]
        :param edge_head: None or [batch, length]
        :param edge_labels: None or [batch, length]
        :param corefs: None or [batch, length]
        :param mask: [batch, length]
        """
        batch_size, _, hidden_size = memory_bank.size()
        head_sentinel = self.head_sentinel.expand([batch_size, 1, hidden_size])
        memory_bank = torch.cat([head_sentinel, memory_bank], 1)
        if edge_heads is not None:
            edge_heads = torch.cat([edge_heads.new_zeros(batch_size, 1), edge_heads], 1)
        if edge_labels is not None:
            edge_labels = torch.cat([edge_labels.new_zeros(batch_size, 1), edge_labels], 1)
        if corefs is not None:
            corefs = torch.cat([corefs.new_zeros(batch_size, 1), corefs], 1)
        mask = torch.cat([mask.new_ones(batch_size, 1), mask], 1)
        return memory_bank, edge_heads, edge_labels, corefs, mask

    def _get_edge_node_scores(self, edge_node_h, edge_node_m, mask):
        edge_node_scores = self.biaffine_attention(edge_node_h, edge_node_m, mask_d=mask, mask_e=mask).squeeze(1)
        return edge_node_scores

    def _get_edge_label_scores(self, edge_label_h, edge_label_m, edge_heads):
        """
        Compute the edge label scores.
        :param edge_label_h: [batch, length, edge_label_hidden_size]
        :param edge_label_m: [batch, length, edge_label_hidden_size]
        :param heads: [batch, length] -- element at [i, j] means the head index of node_j at batch_i.
        :return: [batch, length, num_labels]
        """
        batch_size = edge_label_h.size(0)
        # Create indexing matrix for batch: [batch, 1]
        batch_index = torch.arange(0, batch_size).view(batch_size, 1).type_as(edge_heads.data).long()

        # Select the heads' representations based on the gold/predicted heads.
        # [batch, length, edge_label_hidden_size]
        edge_label_h = edge_label_h[batch_index, edge_heads.data].contiguous()
        edge_label_m = edge_label_m.contiguous()

        # [batch, length, num_labels]
        edge_label_scores = self.edge_label_bilinear(edge_label_h, edge_label_m)

        return edge_label_scores

