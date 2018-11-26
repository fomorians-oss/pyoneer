import collections
import functools

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import random_ops

from trfl import sequence_ops

from pyoneer.features import array_ops as parray_ops


class TensorTuple(object):

    def concat(self, other_rollout, axis=0):
        return self.__class__(
            *[array_ops.concat([
                    getattr(self, field), 
                    getattr(other_rollout, field)], 
                    axis=axis) 
               for field in self._fields])

    def pad_or_truncate(self, size):
        return self.__class__(
            *[parray_ops.pad_or_truncate(getattr(self, field), size, axis=1, pad_value=0) 
               for field in self._fields])


class Rollout(TensorTuple, collections.namedtuple(
        'Rollout', ['states', 'actions', 'rewards', 'weights'])):
    """Holder for states, actions, rewards, and weights."""

    def __len__(self):
        return array_ops.shape(self.states)[0]

    def sample_initial_states(self, size):
        samples = random_ops.random_uniform([size], 0, self.states.shape[0], dtype=dtypes.int32)
        initial_states = array_ops.gather(self.states[:, 0], samples)
        return initial_states

    @functools.lru_cache(maxsize=None)
    def discounted_returns(self, decay):
        sequence = parray_ops.swap_time_major(self.rewards)
        decay = gen_array_ops.broadcast_to(decay, sequence)
        multi_step_returns = sequence_ops.scan_discounted_sum(
            sequence, decay, array_ops.zeros_like(sequence[0]), reverse=True, back_prop=False)
        return parray_ops.swap_time_major(multi_step_returns)
