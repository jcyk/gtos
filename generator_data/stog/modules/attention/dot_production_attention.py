import torch


class DotProductAttention(torch.nn.Module):

    def __init__(self, decoder_hidden_size, encoder_hidden_size, share_linear=True):
        super(DotProductAttention, self).__init__()
        self.decoder_hidden_size = decoder_hidden_size
        self.encoder_hidden_size = encoder_hidden_size
        self.linear_layer = torch.nn.Linear(decoder_hidden_size, encoder_hidden_size, bias=False)
        self.share_linear = share_linear

    def forward(self, decoder_input, encoder_input):
        """
        :param decoder_input:  [batch, decoder_seq_length, decoder_hidden_size]
        :param encoder_input:  [batch, encoder_seq_length, encoder_hidden_size]
        :return:  [batch, decoder_seq_length, encoder_seq_length]
        """
        decoder_input = self.linear_layer(decoder_input)
        if self.share_linear:
            encoder_input = self.linear_layer(encoder_input)

        encoder_input = encoder_input.transpose(1, 2)
        return torch.bmm(decoder_input, encoder_input)
