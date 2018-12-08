from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import random_ops

from trfl import sequence_ops

from pyoneer.manip import array_ops as parray_ops


def tensortuple(*args, **kwargs):
    tcls = collections.namedtuple(*args, **kwargs)

    def concat(self, other_rollout, axis=0):
        """Concatenates tensors by a given axes."""
        return self.__class__(
            *[array_ops.concat([
                    getattr(self, field), 
                    getattr(other_rollout, field)], 
                    axis=axis) 
            for field in self._fields])

    def pad_or_truncate(self, size, axis=1, pad_value=0):
        """Forces tensors to be a certain size by a given axis and optional padding value."""
        return self.__class__(
            *[parray_ops.pad_or_truncate(getattr(self, field), size, axis=axis, pad_value=pad_value) 
            for field in self._fields])

    def tensor_slice(self, key):
        """Slice the tensor with the given `slice(...)`."""
        return self.__class__(
            *[getattr(self, field)[key] for field in self._fields])

    tcls.slice = tensor_slice
    tcls.concat = concat
    tcls.pad_or_truncate = pad_or_truncate
    return tcls


class Rollout(tensortuple('Rollout', ['states', 'actions', 'rewards', 'weights'])):
    """Tuple of Tensors containing (states, actions, rewards, weights)."""

    @property
    @functools.lru_cache(maxsize=None)
    def next_states(self):
        """Compute the next states.

        Returns:
            Tensor of same shape as `states`.
        """
        return parray_ops.shift(self.states, axis=-2, rotations=-1)