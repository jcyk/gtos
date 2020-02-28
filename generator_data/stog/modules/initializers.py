"""
Adopted from AllenNLP:
    https://github.com/allenai/allennlp/blob/v0.6.1/allennlp/nn/initializers.py

An initializer is just a PyTorch function.
Here we implement a proxy class that allows us
to register them and supply any additional function arguments
(for example, the ``mean`` and ``std`` of a normal initializer)
as named arguments to the constructor.
The available initialization functions are
* `"normal" <http://pytorch.org/docs/master/nn.html?highlight=orthogonal#torch.nn.init.normal_>`_
* `"uniform" <http://pytorch.org/docs/master/nn.html?highlight=orthogonal#torch.nn.init.uniform_>`_
* `"constant" <http://pytorch.org/docs/master/nn.html?highlight=orthogonal#torch.nn.init.constant_>`_
* `"eye" <http://pytorch.org/docs/master/nn.html?highlight=orthogonal#torch.nn.init.eye_>`_
* `"dirac" <http://pytorch.org/docs/master/nn.html?highlight=orthogonal#torch.nn.init.dirac_>`_
* `"xavier_uniform" <http://pytorch.org/docs/master/nn.html?highlight=orthogonal#torch.nn.init.xavier_uniform_>`_
* `"xavier_normal" <http://pytorch.org/docs/master/nn.html?highlight=orthogonal#torch.nn.init.xavier_normal_>`_
* `"kaiming_uniform" <http://pytorch.org/docs/master/nn.html?highlight=orthogonal#torch.nn.init.kaiming_uniform_>`_
* `"kaiming_normal" <http://pytorch.org/docs/master/nn.html?highlight=orthogonal#torch.nn.init.kaiming_normal_>`_
* `"orthogonal" <http://pytorch.org/docs/master/nn.html?highlight=orthogonal#torch.nn.init.orthogonal_>`_
* `"sparse" <http://pytorch.org/docs/master/nn.html?highlight=orthogonal#torch.nn.init.sparse_>`_
* :func:`"block_orthogonal" <block_orthogonal>`
* :func:`"uniform_unit_scaling" <uniform_unit_scaling>`
"""
import re
import math
from typing import Callable, List, Tuple, Type, Iterable
import itertools

import torch
import torch.nn.init

from stog.utils import logging
from stog.utils.checks import ConfigurationError

logger = logging.init_logger()  # pylint: disable=invalid-name


def uniform_unit_scaling(tensor: torch.Tensor, nonlinearity: str = "linear"):
    """
    An initaliser which preserves output variance for approximately gaussian
    distributed inputs. This boils down to initialising layers using a uniform
    distribution in the range ``(-sqrt(3/dim[0]) * scale, sqrt(3 / dim[0]) * scale)``, where
    ``dim[0]`` is equal to the input dimension of the parameter and the ``scale``
    is a constant scaling factor which depends on the non-linearity used.
    See `Random Walk Initialisation for Training Very Deep Feedforward Networks
    <https://www.semanticscholar.org/paper/Random-Walk-Initialization-for-Training-Very-Deep-Sussillo-Abbott/be9728a0728b6acf7a485225b1e41592176eda0b>`_
    for more information.
    Parameters
    ----------
    tensor : ``torch.Tensor``, required.
        The tensor to initialise.
    nonlinearity : ``str``, optional (default = "linear")
        The non-linearity which is performed after the projection that this
        tensor is involved in. This must be the name of a function contained
        in the ``torch.nn.functional`` package.
    Returns
    -------
    The initialised tensor.
    """
    size = 1.
    # Estimate the input size. This won't work perfectly,
    # but it covers almost all use cases where this initialiser
    # would be expected to be useful, i.e in large linear and
    # convolutional layers, as the last dimension will almost
    # always be the output size.
    for dimension in list(tensor.size())[:-1]:
        size *= dimension

    activation_scaling = torch.nn.init.calculate_gain(nonlinearity, tensor)
    max_value = math.sqrt(3 / size) * activation_scaling

    return tensor.uniform_(-max_value, max_value)


def block_orthogonal(tensor: torch.Tensor,
                     split_sizes: List[int],
                     gain: float = 1.0) -> None:
    """
    An initializer which allows initializing model parameters in "blocks". This is helpful
    in the case of recurrent models which use multiple gates applied to linear projections,
    which can be computed efficiently if they are concatenated together. However, they are
    separate parameters which should be initialized independently.
    Parameters
    ----------
    tensor : ``torch.Tensor``, required.
        A tensor to initialize.
    split_sizes : List[int], required.
        A list of length ``tensor.ndim()`` specifying the size of the
        blocks along that particular dimension. E.g. ``[10, 20]`` would
        result in the tensor being split into chunks of size 10 along the
        first dimension and 20 along the second.
    gain : float, optional (default = 1.0)
        The gain (scaling) applied to the orthogonal initialization.
    """
    data = tensor.data
    sizes = list(tensor.size())
    if any([a % b != 0 for a, b in zip(sizes, split_sizes)]):
        raise ConfigurationError("tensor dimensions must be divisible by their respective "
                                 "split_sizes. Found size: {} and split_sizes: {}".format(sizes, split_sizes))
    indexes = [list(range(0, max_size, split))
               for max_size, split in zip(sizes, split_sizes)]
    # Iterate over all possible blocks within the tensor.
    for block_start_indices in itertools.product(*indexes):
        # A list of tuples containing the index to start at for this block
        # and the appropriate step size (i.e split_size[i] for dimension i).
        index_and_step_tuples = zip(block_start_indices, split_sizes)
        # This is a tuple of slices corresponding to:
        # tensor[index: index + step_size, ...]. This is
        # required because we could have an arbitrary number
        # of dimensions. The actual slices we need are the
        # start_index: start_index + step for each dimension in the tensor.
        block_slice = tuple([slice(start_index, start_index + step)
                             for start_index, step in index_and_step_tuples])
        data[block_slice] = torch.nn.init.orthogonal_(tensor[block_slice].contiguous(), gain=gain)


def zero(tensor: torch.Tensor) -> None:
    return tensor.data.zero_()

def lstm_hidden_bias(tensor: torch.Tensor) -> None:
    """
    Initialize the biases of the forget gate to 1, and all other gates to 0,
    following Jozefowicz et al., An Empirical Exploration of Recurrent Network Architectures
    """
    # gates are (b_hi|b_hf|b_hg|b_ho) of shape (4*hidden_size)
    tensor.data.zero_()
    hidden_size = tensor.shape[0] // 4
    tensor.data[hidden_size:(2 * hidden_size)] = 1.0
