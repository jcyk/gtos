import torch

from stog.metrics.seq2seq_metrics import Seq2SeqMetrics

class Generator(torch.nn.Module):

    def __init__(self, input_size, vocab_size, pad_idx):
        super(Generator, self).__init__()
        self._generator = torch.nn.Sequential(
            torch.nn.Linear(input_size, vocab_size),
            torch.nn.LogSoftmax(dim=-1)
        )
        self.criterion = torch.nn.NLLLoss(
            ignore_index=pad_idx, reduction='sum'
        )
        self.metrics = Seq2SeqMetrics()
        self.pad_idx = pad_idx

    def forward(self, inputs):
        """Transform inputs to vocab-size space and compute logits.

        :param inputs:  [batch, seq_length, input_size]
        :return:  [batch, seq_length, vocab_size]
        """
        batch_size, seq_length, _ = inputs.size()
        inputs = inputs.view(batch_size * seq_length, -1)
        scores = self._generator(inputs)
        scores = scores.view(batch_size, seq_length, -1)
        _, predictions = scores.max(2)
        return dict(
            scores=scores,
            predictions=predictions
        )

    def compute_loss(self, inputs, targets):
        batch_size, seq_length, _ = inputs.size()
        output = self(inputs)
        scores = output['scores'].view(batch_size * seq_length, -1)
        predictions = output['predictions'].view(-1)
        targets = targets.view(-1)

        loss = self.criterion(scores, targets)

        non_pad = targets.ne(self.pad_idx)
        num_correct = predictions.eq(targets).masked_select(non_pad).sum().item()
        num_non_pad = non_pad.sum().item()
        self.metrics(loss.item(), num_non_pad, num_correct)

        return dict(
            loss=loss.div(float(num_non_pad)),
            predictions=output['predictions']
        )

    @classmethod
    def from_params(cls, params):
        return cls(
            input_size=params['input_size'],
            vocab_size=params['vocab_size'],
            pad_idx=params['pad_idx']
        )
