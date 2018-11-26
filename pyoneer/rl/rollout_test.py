from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.eager import context
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test

from pyoneer.rl import rollout_impl


def create_rollout(episodes, timesteps, state_dims):
    return rollout_impl.Rollout(
        states=array_ops.zeros([episodes, timesteps, state_dims], dtype=dtypes.float32),
        actions=array_ops.zeros([episodes, timesteps], dtype=dtypes.float32),
        rewards=array_ops.zeros([episodes, timesteps], dtype=dtypes.float32),
        weights=array_ops.zeros([episodes, timesteps], dtype=dtypes.float32))


class RolloutTest(test.TestCase):

    def test_rollout_concat_batch(self):
        with context.eager_mode():
            rollouts = create_rollout(2, 2, 4)
            other_rollouts = create_rollout(3, 2, 4)
            actual_rollouts = rollouts.concat(other_rollouts, axis=0)
            expected_rollouts = create_rollout(5, 2, 4)
            self.assertAllEqual(actual_rollouts.states, expected_rollouts.states)
    
    def test_rollout_concat_time(self):
        with context.eager_mode():
            rollouts = create_rollout(2, 2, 4)
            other_rollouts = create_rollout(2, 3, 4)
            actual_rollouts = rollouts.concat(other_rollouts, axis=1)
            expected_rollouts = create_rollout(2, 5, 4)
            self.assertAllEqual(actual_rollouts.states, expected_rollouts.states)

    def test_rollout_sample_initial_states(self):
        with context.eager_mode():
            rollouts = create_rollout(2, 5, 4)
            actual_states = rollouts.sample_initial_states(5)
            expected_states_shape = [5, 4]
            self.assertAllEqual(array_ops.shape(actual_states), expected_states_shape)

    def test_rollout_pad_or_truncate(self):
        with context.eager_mode():
            rollouts = create_rollout(2, 5, 4)
            actual_rollouts = rollouts.pad_or_truncate(10)
            expected_rollouts = create_rollout(2, 10, 4)
            self.assertAllEqual(actual_rollouts.states, expected_rollouts.states)


if __name__ == '__main__':
    test.main()