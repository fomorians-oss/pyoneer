from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.eager import context
from tensorflow.python.platform import test

from pyoneer.rl.targets import target_ops


class TargetOpsTest(test.TestCase):
    def test_discounted_rewards(self):
        with context.eager_mode():
            rewards = tf.constant([[0.0, 0.0, 1.0]])
            outputs = target_ops.discounted_rewards(
                rewards, discount_factor=0.99, weights=1.0)
            expected = tf.constant([[0.9801, 0.99, 1.0]])
            self.assertAllClose(outputs, expected)

    def test_generalized_advantage_estimate(self):
        with context.eager_mode():
            rewards = tf.constant([[0.0, 0.0, 1.0]])
            values = tf.constant([[0.0, 0.0, 1.0]])
            outputs = target_ops.generalized_advantages(
                rewards,
                values,
                discount_factor=0.99,
                lambda_factor=0.95,
                weights=1.0,
                normalize=True)
            expected = tf.constant([[0.564769, 0.840459, -1.405228]])
            self.assertAllClose(outputs, expected)


if __name__ == '__main__':
    test.main()
