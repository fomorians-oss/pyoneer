from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.eager import context
from tensorflow.python.platform import test

from pyoneer.rl.losses.policy_gradient_impl import (
    policy_gradient_loss, clipped_policy_gradient_loss)


class PolicyGradientTest(test.TestCase):
    def test_policy_gradient_loss(self):
        with context.eager_mode():
            log_probs = tf.log(tf.constant([0.9, 0.8, 0.8, 0.8]))
            advantages = tf.constant([1.0, 0.0, 1.0, 0.0])
            weights = tf.constant([1.0, 0.0, 1.0, 0.0])

            loss = policy_gradient_loss(
                log_probs=log_probs, advantages=advantages, weights=weights)
            expected = tf.losses.compute_weighted_loss(
                -log_probs * advantages, weights=weights)

            self.assertAllClose(loss, expected)

    def test_clipped_policy_gradient_loss(self):
        with context.eager_mode():
            epsilon_clipping = 0.2

            log_probs = tf.log(tf.constant([0.9, 0.8, 0.8, 0.8]))
            log_probs_anchor = tf.log(tf.constant([0.95, 0.85, 0.85, 0.85]))

            advantages = tf.constant([1.0, 0.0, 1.0, 0.0])
            weights = tf.constant([1.0, 0.0, 1.0, 0.0])

            # using the function
            loss = clipped_policy_gradient_loss(
                log_probs=log_probs,
                log_probs_anchor=log_probs_anchor,
                advantages=advantages,
                epsilon_clipping=epsilon_clipping,
                weights=weights)

            # manual check
            ratio = tf.exp(log_probs - log_probs_anchor)
            surrogate1 = ratio * advantages
            surrogate2 = tf.clip_by_value(ratio, 1 - epsilon_clipping,
                                          1 + epsilon_clipping) * advantages

            surrogate_min = tf.minimum(surrogate1, surrogate2)

            expected = -tf.losses.compute_weighted_loss(
                losses=surrogate_min, weights=weights)

            self.assertAllClose(loss, expected)


if __name__ == "__main__":
    test.main()
