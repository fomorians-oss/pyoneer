from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow_probability as tfp
import tensorflow as tf

from tensorflow.python.eager import context
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test

from trfl import policy_gradient_ops

from pyoneer.losses import vanilla_policy_gradient_loss
from pyoneer.losses import entropy_loss


class LossTest(test.TestCase):
    def test_vanilla_policy_gradient_loss(self):
        with context.eager_mode():

            probs = array_ops.constant([.9, .8, .8, .8])
            log_probs = math_ops.log(probs)
            advantages = array_ops.constant([1., 0., 1., 0.])

            loss = vanilla_policy_gradient_loss(log_probs, advantages)
            expected = tf.losses.compute_weighted_loss(-log_probs * advantages)

            self.assertAllClose(loss, expected)

    def test_weighted_vanilla_policy_gradient_loss(self):
        with context.eager_mode():

            probs = array_ops.constant([.9, .8, .8, .8])
            log_probs = math_ops.log(probs)
            advantages = array_ops.constant([1., 0., 1., 0.])
            weights = array_ops.constant([1., 0., 1., 0.])

            loss = vanilla_policy_gradient_loss(log_probs, advantages, weights)
            expected = tf.losses.compute_weighted_loss(
                -log_probs * advantages, weights=weights)

            self.assertAllClose(loss, expected)

    def test_vanilla_policy_entropy_loss(self):
        with context.eager_mode():
            entropy = tf.constant([0., 0., 1., 1.])
            loss = entropy_loss(entropy=entropy)
            expected = -tf.losses.compute_weighted_loss(losses=entropy)

            self.assertAllClose(loss, expected)

    def test_weighted_vanilla_policy_entropy_loss(self):
        with context.eager_mode():
            weights = tf.constant([0., 1., 0., 1.])
            entropy = tf.constant([0., 0., 1., 1.])
            loss = entropy_loss(entropy=entropy, weights=weights)

            expected = -tf.losses.compute_weighted_loss(
                losses=entropy, weights=weights)

            self.assertAllClose(loss, expected)


if __name__ == "__main__":
    test.main()
