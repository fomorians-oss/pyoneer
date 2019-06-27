from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.keras.utils import losses_utils

from pyoneer.rl.losses.policy_gradient_ops import (
    PolicyGradient,
    ClippedPolicyGradient,
    PolicyEntropy,
)


class PolicyGradientTest(tf.test.TestCase):
    def test_policy_gradient(self):
        log_probs = tf.math.log(tf.constant([0.9, 0.8, 0.8, 0.8]))
        advantages = tf.constant([1.0, 0.0, 1.0, 0.0])
        sample_weight = tf.constant([1.0, 0.0, 1.0, 0.0])

        loss_fn = PolicyGradient()
        loss = loss_fn(
            log_probs=log_probs, advantages=advantages, sample_weight=sample_weight
        )
        expected = losses_utils.compute_weighted_loss(
            -log_probs * advantages, sample_weight=sample_weight
        )

        self.assertAllClose(loss, expected)

    def test_clipped_policy_gradient(self):
        epsilon_clipping = 0.2

        log_probs = tf.math.log(tf.constant([0.9, 0.8, 0.8, 0.8]))
        log_probs_anchor = tf.math.log(tf.constant([0.95, 0.85, 0.85, 0.85]))

        advantages = tf.constant([1.0, 0.0, 1.0, 0.0])
        sample_weight = tf.constant([1.0, 0.0, 1.0, 0.0])

        # using the function
        loss_fn = ClippedPolicyGradient(epsilon_clipping=epsilon_clipping)
        loss = loss_fn(
            log_probs=log_probs,
            log_probs_anchor=log_probs_anchor,
            advantages=advantages,
            sample_weight=sample_weight,
        )

        # manual check
        ratio = tf.exp(log_probs - log_probs_anchor)
        surrogate1 = ratio * advantages
        surrogate2 = (
            tf.clip_by_value(ratio, 1 - epsilon_clipping, 1 + epsilon_clipping)
            * advantages
        )

        surrogate_min = tf.minimum(surrogate1, surrogate2)

        expected = -losses_utils.compute_weighted_loss(
            losses=surrogate_min, sample_weight=sample_weight
        )

        self.assertAllClose(loss, expected)

    def test_entropy(self):
        entropy = tf.constant([0.2, 0.3, 0.4])
        expected = -tf.constant(0.3)
        loss_fn = PolicyEntropy()
        losses = loss_fn(entropy)
        self.assertAllClose(losses, expected)


if __name__ == "__main__":
    tf.test.main()
