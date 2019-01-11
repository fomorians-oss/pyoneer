from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.eager import context
from tensorflow.python.platform import test

from pyoneer.losses import proximal_policy_ratio_gradient_loss


class LossTest(test.TestCase):
    def test_ppo_gradient_loss(self):
        with context.eager_mode():

            epsilon_clipping = 0.2

            probs = tf.constant([.9, .8, .8, .8])
            log_probs = tf.log(probs)

            probs_anchor = tf.constant([.95, .85, .85, .85])
            log_probs_anchor = tf.log(probs_anchor)

            advantages = tf.constant([1., 0., 1., 0.])

            # using the function
            loss = proximal_policy_ratio_gradient_loss(
                log_probs=log_probs,
                log_probs_anchor=log_probs_anchor,
                epsilon_clipping=epsilon_clipping,
                advantages=advantages)

            # manual check
            ratio = tf.exp(log_probs - log_probs_anchor)
            surrogate1 = ratio * advantages
            surrogate2 = tf.clip_by_value(ratio, 1 - epsilon_clipping,
                                          1 + epsilon_clipping) * advantages

            surrogate_min = tf.minimum(surrogate1, surrogate2)

            expected = -tf.losses.compute_weighted_loss(losses=surrogate_min)

            self.assertAllClose(loss, expected)

    def test_ppo_weighted_gradient_loss(self):
        with context.eager_mode():

            epsilon_clipping = 0.2
            weights = tf.constant([1, 1, 0, 0])

            probs = tf.constant([.9, .8, .8, .8])
            log_probs = tf.log(probs)

            probs_anchor = tf.constant([.95, .85, .85, .85])
            log_probs_anchor = tf.log(probs_anchor)

            advantages = tf.constant([1., 0., 1., 0.])

            # using the function
            loss = proximal_policy_ratio_gradient_loss(
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
