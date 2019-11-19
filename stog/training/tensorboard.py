from typing import Any

import torch
from tensorboardX import SummaryWriter


class TensorboardWriter:
    """
    Wraps a pair of ``SummaryWriter`` instances but is a no-op if they're ``None``.
    Allows Tensorboard logging without always checking for Nones first.
    """
    def __init__(self, train_log=None, dev_log=None) -> None:
        self._train_log = SummaryWriter(train_log) if train_log is not None else None
        self._dev_log = SummaryWriter(dev_log) if dev_log is not None else None

    @staticmethod
    def _item(value: Any):
        if hasattr(value, 'item'):
            val = value.item()
        else:
            val = value
        return val

    def add_train_scalar(self, name: str, value: float, global_step: int) -> None:
        # get the scalar
        if self._train_log is not None:
            self._train_log.add_scalar(name, self._item(value), global_step)

    def add_train_histogram(self, name: str, values: torch.Tensor, global_step: int) -> None:
        if self._train_log is not None:
            if isinstance(values, torch.Tensor):
                values_to_write = values.cpu().data.numpy().flatten()
                self._train_log.add_histog.am(name, values_to_write, global_step)

    def add_dev_scalar(self, name: str, value: float, global_step: int) -> None:

        if self._dev_log is not None:
            self._dev_log.add_scalar(name, self._item(value), global_step)

