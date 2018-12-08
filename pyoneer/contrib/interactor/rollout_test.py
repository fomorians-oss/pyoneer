from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.eager import context
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test

from pyoneer.contrib.interactor import rollout_impl


def create_rollout(episodes, timesteps, state_dims):
    return rollout_impl.Rollout(
        states=array_ops.zeros([episodes, timesteps, state_dims], dtype=dtypes.float32),
        actions=array_ops.zeros([episodes, timesteps], dtype=dtypes.float32),
        rewards=array_ops.zeros([episodes, timesteps], dtype=dtypes.float32),
        weights=array_ops.zeros([episodes, timesteps], dtype=dtypes.float32))


class RolloutTest(test.TestCase):

    def testRolloutConcatBatch(self):
        with context.eager_mode():
            rollouts = create_rollout(2, 2, 4)
            other_rollouts = create_rollout(3, 2, 4)
            actual_rollouts = rollouts.concat(other_rollouts, axis=0)
            expected_rollouts = create_rollout(5, 2, 4)
            self.assertAllEqual(actual_rollouts.states, expected_rollouts.states)
    
    def testRolloutConcatTime(self):
        with context.eager_mode():
            rollouts = create_rollout(2, 2, 4)
            other_rollouts = create_rollout(2, 3, 4)
            actual_rollouts = rollouts.concat(other_rollouts, axis=1)
            expected_rollouts = create_rollout(2, 5, 4)
            self.assertAllEqual(actual_rollouts.states, expected_rollouts.states)

    def testRolloutPadOrTruncate(self):
        with context.eager_mode():
            rollouts = create_rollout(2, 5, 4)
            actual_rollouts = rollouts.pad_or_truncate(10)
            expected_rollouts = create_rollout(2, 10, 4)
            self.assertAllEqual(actual_rollouts.states, expected_rollouts.states)

    def testRolloutNextStatesContiguous(self):
        with context.eager_mode():
            actual_rollouts = rollout_impl.Rollout(
                states=array_ops.constant([[[0], [1], [2], [3]]]),
                actions=array_ops.constant([[0, 1, 2, 3]]),
                rewards=array_ops.constant([[1, 1, 1, 1]]),
                weights=array_ops.constant([[1, 1, 1, 0]]))
            expected_next_states = array_ops.constant([[[1], [2], [3], [0]]])
            self.assertAllEqual(actual_rollouts.next_states, expected_next_states)

    def testRolloutNextStatesNonContiguous(self):
        with context.eager_mode():
            actual_rollouts = rollout_impl.Rollout(
                states=array_ops.constant([[0], [1], [2], [3]]),
                actions=array_ops.constant([0, 1, 2, 3]),
                rewards=array_ops.constant([1, 1, 1, 1]),
                weights=array_ops.constant([1, 1, 1, 0]))
            expected_next_states = array_ops.constant([[1], [2], [3], [0]])
            self.assertAllEqual(actual_rollouts.next_states, expected_next_states)


if __name__ == '__main__':
    test.main()