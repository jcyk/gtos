import copy

import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RNNDecoderBase(torch.nn.Module):

    def __init__(self, rnn_cell, dropout):
        super(RNNDecoderBase, self).__init__()
        self.rnn_cell = rnn_cell
        self.dropout = dropout

    def forward(self, *input):
        raise NotImplementedError


class InputFeedRNNDecoder(RNNDecoderBase):

    def __init__(self,
                 rnn_cell,
                 dropout,
                 attention_layer,
                 source_copy_attention_layer=None,
                 coref_attention_layer=None,
                 use_coverage=False):
        super(InputFeedRNNDecoder, self).__init__(rnn_cell, dropout)
        self.attention_layer = attention_layer
        self.source_copy_attention_layer = source_copy_attention_layer
        self.coref_attention_layer = coref_attention_layer
        self.use_coverage = use_coverage

    def forward(self, inputs, memory_bank, mask, hidden_state,
                input_feed=None, target_copy_hidden_states=None, coverage=None):
        """

        :param inputs: [batch_size, decoder_seq_length, embedding_size]
        :param memory_bank: [batch_size, encoder_seq_length, encoder_hidden_size]
        :param mask:  None or [batch_size, decoder_seq_length]
        :param hidden_state: a tuple of (state, memory) with shape [num_encoder_layers, batch_size, encoder_hidden_size]
        :param input_feed: None or [batch_size, 1, hidden_size]
        :param target_copy_hidden_states: None or [batch_size, seq_length, hidden_size]
        :param coverage: None or [batch_size, 1, encode_seq_length]
        :return:
        """
        batch_size, sequence_length, _ = inputs.size()
        one_step_length = [1] * batch_size
        source_copy_attentions = []
        target_copy_attentions = []
        coverage_records = []
        decoder_hidden_states = []
        rnn_hidden_states = []

        if input_feed is None:
            input_feed = inputs.new_zeros(batch_size, 1, self.rnn_cell.hidden_size)

        if target_copy_hidden_states is None:
            target_copy_hidden_states = []

        if self.use_coverage and coverage is None:
            coverage = inputs.new_zeros(batch_size, 1, memory_bank.size(1))

        for step_i, input in enumerate(inputs.split(1, dim=1)):
            # input: [batch_size, 1, embeddings_size]
            # input_feed: [batch_size, 1, hidden_size]
            _input = torch.cat([input, input_feed], 2)
            packed_input = pack_padded_sequence(_input, one_step_length, batch_first=True)
            # hidden_state: a tuple of (state, memory) with shape [num_layers, batch_size, hidden_size]
            packed_output, hidden_state = self.rnn_cell(packed_input, hidden_state)
            # output: [batch_size, 1, hidden_size]
            output, _ = pad_packed_sequence(packed_output, batch_first=True)
            rnn_hidden_states.append(output)

            coverage_records.append(coverage)
            output, std_attention, coverage = self.attention_layer(
                output, memory_bank, mask, coverage)

            output = self.dropout(output)
            input_feed = output

            if self.source_copy_attention_layer is not None:
                _, source_copy_attention = self.source_copy_attention_layer(
                    output, memory_bank, mask)
                source_copy_attentions.append(source_copy_attention)
            else:
                source_copy_attentions.append(std_attention)

            if self.coref_attention_layer is not None:
                if len(target_copy_hidden_states) == 0:
                    target_copy_attention = inputs.new_zeros(batch_size, 1, sequence_length)

                else:
                    target_copy_memory = torch.cat(target_copy_hidden_states, 1)

                    if sequence_length == 1:
                        _, target_copy_attention, _ = self.coref_attention_layer(
                            output, target_copy_memory)
                    else:
                        _, target_copy_attention, _ = self.coref_attention_layer(
                            output, target_copy_memory)
                        target_copy_attention = torch.nn.functional.pad(
                            target_copy_attention, (0, sequence_length - step_i), 'constant', 0
                        )

                target_copy_attentions.append(target_copy_attention)

            target_copy_hidden_states.append(output)
            decoder_hidden_states.append(output)

        decoder_hidden_states = torch.cat(decoder_hidden_states, 1)
        rnn_hidden_states = torch.cat(rnn_hidden_states, 1)
        source_copy_attentions = torch.cat(source_copy_attentions, 1)
        if len(target_copy_attentions):
            target_copy_attentions = torch.cat(target_copy_attentions, 1)
        else:
            target_copy_attentions = None
        if self.use_coverage:
            coverage_records = torch.cat(coverage_records, 1)
        else:
            coverage_records = None

        return dict(
            decoder_hidden_states=decoder_hidden_states,
            rnn_hidden_states=rnn_hidden_states,
            source_copy_attentions=source_copy_attentions,
            target_copy_attentions=target_copy_attentions,
            coverage_records=coverage_records,
            last_hidden_state=hidden_state,
            input_feed=input_feed,
            coverage=coverage
        )
