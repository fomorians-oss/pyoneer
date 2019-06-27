from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from pyoneer.rl.targets.target_ops import DiscountedReturns, GeneralizedAdvantages


class TargetOpsTest(tf.test.TestCase):
    def test_discounted_returns(self):
        rewards = tf.constant([[0.0, 0.0, 1.0]])
        discounted_rewards = DiscountedReturns(discount_factor=0.99)
        outputs = discounted_rewards(rewards, sample_weight=1.0)
        expected = tf.constant([[0.9801, 0.99, 1.0]])
        self.assertAllClose(outputs, expected)

    def test_generalized_advantages(self):
        rewards = tf.constant([[0.0, 0.0, 1.0]])
        values = tf.constant([[0.0, 0.0, 1.0]])
        generalized_advantages = GeneralizedAdvantages(
            discount_factor=0.99, lambda_factor=0.95, normalize=True
        )
        outputs = generalized_advantages(rewards, values, sample_weight=1.0)
        expected = tf.constant([[0.564769, 0.840459, -1.405228]])
        self.assertAllClose(outputs, expected)


if __name__ == "__main__":
    tf.test.main()
