from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.eager import context
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test

from pyoneer.rl import rollout_impl


def create_contiguous_rollout(episodes, timesteps, state_dims):
    return rollout_impl.ContiguousRollout(
        states=array_ops.zeros([episodes, timesteps, state_dims], dtype=dtypes.float32),
        actions=array_ops.zeros([episodes, timesteps], dtype=dtypes.float32),
        rewards=array_ops.zeros([episodes, timesteps], dtype=dtypes.float32),
        weights=array_ops.zeros([episodes, timesteps], dtype=dtypes.float32))


def create_rollout(episodes, state_dims):
    return rollout_impl.Rollout(
        states=array_ops.zeros([episodes, state_dims], dtype=dtypes.float32),
        next_states=array_ops.zeros([episodes, state_dims], dtype=dtypes.float32),
        actions=array_ops.zeros([episodes], dtype=dtypes.float32),
        rewards=array_ops.zeros([episodes], dtype=dtypes.float32),
        weights=array_ops.zeros([episodes], dtype=dtypes.float32))


class ContiguousRolloutTest(test.TestCase):

    def test_contignuous_rollout_concat_batch(self):
        with context.eager_mode():
            rollouts = create_contiguous_rollout(2, 2, 4)
            other_rollouts = create_contiguous_rollout(3, 2, 4)
            actual_rollouts = rollouts.concat(other_rollouts, axis=0)
            expected_rollouts = create_contiguous_rollout(5, 2, 4)
            self.assertAllEqual(actual_rollouts.states, expected_rollouts.states)
    
    def test_contignuous_rollout_concat_time(self):
        with context.eager_mode():
            rollouts = create_contiguous_rollout(2, 2, 4)
            other_rollouts = create_contiguous_rollout(2, 3, 4)
            actual_rollouts = rollouts.concat(other_rollouts, axis=1)
            expected_rollouts = create_contiguous_rollout(2, 5, 4)
            self.assertAllEqual(actual_rollouts.states, expected_rollouts.states)

    def test_contignuous_rollout_pad_or_truncate(self):
        with context.eager_mode():
            rollouts = create_contiguous_rollout(2, 5, 4)
            actual_rollouts = rollouts.pad_or_truncate(10)
            expected_rollouts = create_contiguous_rollout(2, 10, 4)
            self.assertAllEqual(actual_rollouts.states, expected_rollouts.states)

    def test_contignuous_rollout_next_states(self):
        with context.eager_mode():
            actual_rollouts = rollout_impl.ContiguousRollout(
                states=array_ops.constant([[[0], [1], [2], [3]]]),
                actions=array_ops.constant([[0, 1, 2, 3]]),
                rewards=array_ops.constant([[1, 1, 1, 1]]),
                weights=array_ops.constant([[1, 1, 1, 0]]))
            expected_next_states = array_ops.constant([[[1], [2], [3], [0]]])
            self.assertAllEqual(actual_rollouts.next_states, expected_next_states)


class RolloutTest(test.TestCase):

    def test_rollout_folded(self):
        with context.eager_mode():
            rollouts = create_contiguous_rollout(2, 2, 4)
            actual_rollouts = rollouts.folded
            expected_rollouts = create_rollout(2 * 2, 4)
            self.assertAllEqual(actual_rollouts.states, expected_rollouts.states)


if __name__ == '__main__':
    test.main()