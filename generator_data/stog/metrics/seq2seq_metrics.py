"""Sequence-to-sequence metrics"""
import math

from overrides import overrides

from stog.metrics.metric import Metric


class Seq2SeqMetrics(Metric):
    """
    Accumulator for loss statistics.
    Currently calculates:
    * accuracy
    * perplexity
    * elapsed time
    """

    def __init__(self, loss=0, n_words=0, n_correct=0,
                 n_source_copies=0, n_correct_source_copies=0, n_correct_source_points=0,
                 n_target_copies=0, n_correct_target_copies=0, n_correct_target_points=0
                 ):
        self.loss = loss
        self.n_words = n_words
        self.n_correct = n_correct

        self.n_source_copies = n_source_copies
        self.n_correct_source_copies = n_correct_source_copies
        self.n_correct_source_points = n_correct_source_points

        self.n_target_copies = n_target_copies
        self.n_correct_target_copies = n_correct_target_copies
        self.n_correct_target_points = n_correct_target_points

    def __call__(self, loss, n_words, n_correct,
                 n_source_copies=0, n_correct_source_copies=0, n_correct_source_points=0,
                 n_target_copies=0, n_correct_target_copies=0, n_correct_target_points=0
                 ):
        """
        Update statistics by suming values with another `Statistics` object
        """
        self.loss += loss
        self.n_words += n_words
        self.n_correct += n_correct
        self.n_source_copies += n_source_copies
        self.n_correct_source_copies += n_correct_source_copies
        self.n_correct_source_points += n_correct_source_points
        self.n_target_copies += n_target_copies
        self.n_correct_target_copies += n_correct_target_copies
        self.n_correct_target_points += n_correct_target_points

    def accuracy(self):
        """ compute accuracy """
        return 100 * (self.n_correct / self.n_words)

    def xent(self):
        """ compute cross entropy """
        return self.loss / self.n_words

    def ppl(self):
        """ compute perplexity """
        return math.exp(min(self.loss / self.n_words, 100))

    def copy_accuracy(self, n_correct, n_copies):
        if n_copies == 0:
            return -1
        else:
            return 100 * (n_correct / n_copies)

    def get_metric(self, reset: bool = False):
        metrics = dict(
            all_acc=self.accuracy(),
            src_acc=self.copy_accuracy(self.n_correct_source_copies, self.n_source_copies),
            tgt_acc=self.copy_accuracy(self.n_correct_target_copies, self.n_target_copies),
            # bina_acc=self.binary_accuracy(),
            # xent=self.xent(),
            ppl=self.ppl()
        )
        if reset:
            self.reset()
        return metrics

    @overrides
    def reset(self):
        self.loss = 0
        self.n_words = 0
        self.n_correct = 0
        self.n_target_copies = 0
        self.n_correct_target_copies= 0
        self.n_correct_target_points = 0
        self.n_source_copies = 0
        self.n_correct_source_copies = 0
        self.n_correct_source_points = 0
