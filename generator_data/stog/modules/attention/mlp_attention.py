import torch


class MLPAttention(torch.nn.Module):

    def __init__(self, decoder_hidden_size, encoder_hidden_size, attention_hidden_size, coverage=False, use_concat=False):
        super(MLPAttention, self).__init__()
        self.hidden_size = attention_hidden_size
        self.query_linear = torch.nn.Linear(decoder_hidden_size, self.hidden_size, bias=True)
        self.context_linear = torch.nn.Linear(encoder_hidden_size, self.hidden_size, bias=False)
        self.output_linear = torch.nn.Linear(self.hidden_size, 1, bias=False)
        if coverage:
            self.coverage_linear = torch.nn.Linear(1, self.hidden_size, bias=False)
        self.use_concat = use_concat
        if self.use_concat:
            self.concat_linear = torch.nn.Linear(
                decoder_hidden_size, self.hidden_size, bias=False)

    def forward(self, decoder_input, encoder_input, coverage=None):
        """
        :param decoder_input:  [batch, decoder_seq_length, decoder_hidden_size]
        :param encoder_input:  [batch, encoder_seq_length, encoder_hidden_size]
        :param coverage: [batch, encoder_seq_length]
        :return:  [batch, decoder_seq_length, encoder_seq_length]
        """
        batch_size, decoder_seq_length, decoder_hidden_size = decoder_input.size()
        batch_size, encoder_seq_length, encoder_hidden_size = encoder_input.size()

        decoder_features = self.query_linear(decoder_input)
        decoder_features = decoder_features.unsqueeze(2).expand(
            batch_size, decoder_seq_length, encoder_seq_length, self.hidden_size)
        encoder_features = self.context_linear(encoder_input)
        encoder_features = encoder_features.unsqueeze(1).expand(
            batch_size, decoder_seq_length, encoder_seq_length, self.hidden_size)
        attn_features = decoder_features + encoder_features

        if coverage is not None:
            coverage_features = self.coverage_linear(
                coverage.view(batch_size, 1, encoder_seq_length, 1)).expand(
                batch_size, decoder_seq_length, encoder_seq_length, self.hidden_size)
            attn_features = attn_features + coverage_features

        if self.use_concat:
            # concat_input = torch.cat([
            #     decoder_input.unsqueeze(2).expand(
            #         batch_size, decoder_seq_length, encoder_seq_length, decoder_hid# den_size),
            #     encoder_input.unsqueeze(1).expand(
            #         batch_size, decoder_seq_length, encoder_seq_length, encoder_hidden_size)
            # ], dim=3)
            concat_input = (decoder_input.unsqueeze(2).expand(
                batch_size, decoder_seq_length, encoder_seq_length, decoder_hidden_size) *
                            encoder_input.unsqueeze(1).expand(
                batch_size, decoder_seq_length, encoder_seq_length, encoder_hidden_size))
            concat_features = self.concat_linear(concat_input)
            attn_features = attn_features + concat_features

        e = torch.tanh(attn_features)
        scores = self.output_linear(e).squeeze(3)
        return scores
