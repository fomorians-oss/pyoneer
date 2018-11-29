import collections
import functools

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import random_ops

from trfl import sequence_ops

from pyoneer.manip import array_ops as parray_ops


def tensortuple(*args, **kwargs):
    tcls = collections.namedtuple(*args, **kwargs)

    def tensor_size(self):
        """Shared size of the tensors.
        """
        return array_ops.shape(self._fields[0])[0]

    def folded(self):
        """Forces tensors to be packed into 1 or 2 dimensions.
        """
        return self.__class__(
            *[parray_ops.flatten(getattr(self, field)) for field in self._fields])

    def concat(self, other_rollout, axis=0):
        """Concatenates tensors by a given axes.
        """
        return self.__class__(
            *[array_ops.concat([
                    getattr(self, field), 
                    getattr(other_rollout, field)], 
                    axis=axis) 
            for field in self._fields])

    def pad_or_truncate(self, size, axis=1, pad_value=0):
        """Forces tensors to be a certain size by a given axis and optional padding value
        """
        return self.__class__(
            *[parray_ops.pad_or_truncate(getattr(self, field), size, axis=axis, pad_value=pad_value) 
            for field in self._fields])

    tcls.__len__ = tensor_size
    tcls.folded = property(folded)
    tcls.concat = concat
    tcls.pad_or_truncate = pad_or_truncate
    return tcls


class Rollout(tensortuple('Rollout', ['states', 'next_states', 'actions', 'rewards', 'weights'])):
    """Tuple of Tensors containing [states, next_states, actions, rewards, and weights]."""
    pass


class ContiguousRollout(tensortuple('ContiguousRollout', ['states', 'actions', 'rewards', 'weights'])):
    """Tuple of Tensors containing [states, actions, rewards, and weights]."""

    @property
    def folded(self):
        """Turns a `ContiguousRollout` to a `Rollout`.
        """
        states = self.states
        actions = self.actions
        next_states = self.next_states
        rewards = self.rewards
        weights = self.weights
        rollout = Rollout(
            states=states, 
            next_states=next_states,
            actions=actions,
            rewards=rewards,
            weights=weights)
        return rollout.folded

    @property
    @functools.lru_cache(maxsize=None)
    def next_states(self):
        """Compute the next states from a rollouts.
        """
        return parray_ops.shift(self.states, axis=-2, rotations=-1)

    @functools.lru_cache(maxsize=None)
    def discounted_returns(self, decay):
        """Compute the discounted returns given the decay factor.
        """
        sequence = parray_ops.swap_time_major(self.rewards)
        decay = gen_array_ops.broadcast_to(decay, sequence)
        multi_step_returns = sequence_ops.scan_discounted_sum(
            sequence, decay, array_ops.zeros_like(sequence[0]), reverse=True, back_prop=False)
        return parray_ops.swap_time_major(multi_step_returns)
