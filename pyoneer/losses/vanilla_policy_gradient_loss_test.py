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
from pyoneer.losses import vanilla_policy_entropy_loss


class _TestPolicy(tf.keras.Model):

    def __init__(self, action_size):
        super(_TestPolicy, self).__init__()
        self.linear = tf.layers.Dense(action_size)

    @property
    def trainable_variables(self):
        return None

    def call(self, inputs, training=False, reset_state=True):
        return tfp.distributions.MultivariateNormalDiag(self.linear(inputs))


class LossTest(test.TestCase):

    def test_vanilla_policy_gradient_loss(self):
        with context.eager_mode():

            probs = array_ops.constant([.9, .8, .8, .8])
            log_probs = math_ops.log(probs)
            advantages = array_ops.constant([1.,0.,1.,0.])

            loss = vanilla_policy_gradient_loss(log_probs, advantages)
            expected = tf.losses.compute_weighted_loss(advantages * -log_probs)

            self.assertAllClose(loss, expected)

    def test_weighted_vanilla_policy_gradient_loss(self):
        with context.eager_mode():

            probs = array_ops.constant([.9, .8, .8, .8])
            log_probs = math_ops.log(probs)
            advantages = array_ops.constant([1.,0.,1.,0.])
            weights = array_ops.constant([1.,0.,1.,0.])

            loss = vanilla_policy_gradient_loss(log_probs, 
                                                advantages, 
                                                weights)
            expected = tf.losses.compute_weighted_loss(advantages * -log_probs,
                                             weights=weights)

            self.assertAllClose(loss, expected)

    def test_vanilla_policy_entropy_loss(self):
        with context.eager_mode():
            policy = _TestPolicy(5)

            trained_policy = policy(tf.constant([[0.,0.,1.,1.,1.], 
                                                 [1.,1.,1.,0.,0.]]),
                              training=True)

            entropy_scale = 1.0

            loss = vanilla_policy_entropy_loss(policy=trained_policy,
                                               entropy_scale=entropy_scale)

            intermediate_losses = policy_gradient_ops.policy_entropy_loss(
                trained_policy,
                lambda policies: entropy_scale).loss
            expected = tf.losses.compute_weighted_loss(intermediate_losses)

            self.assertAllClose(loss, expected)
    
    def test_weighted_vanilla_policy_entropy_loss(self):
        with context.eager_mode():
            policy = _TestPolicy(5)
            entropy_scale = 1.0
            weights = tf.constant([0., 1.])

            trained_policy = policy(tf.constant([[0.,0.,1.,1.,1.], 
                                                 [1.,1.,1.,0.,0.]]),
                              training=True)

            loss = vanilla_policy_entropy_loss(policy=trained_policy,
                                               entropy_scale=entropy_scale,
                                               weights=weights)

            intermediate_losses = policy_gradient_ops.policy_entropy_loss(
                trained_policy,
                policy.trainable_variables,
                lambda policies: entropy_scale).loss
            expected = tf.losses.compute_weighted_loss(intermediate_losses, weights=weights)

            self.assertAllClose(loss, expected)

if __name__ == "__main__":
    test.main()
