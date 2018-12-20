from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def vanilla_policy_gradient_loss(log_probs,
                                 advantages,
                                 weights=1.0):
    """ Computes the Vanilla PG loss.
      Args:
      log_prob: 
      advantages:
      weights:
    Returns tensor 
    """

    advantages = tf.stop_gradient(advantages)
    weights = tf.convert_to_tensor(weights)

    unweighted_loss = advantages * -log_probs
    unweighted_loss = tf.check_numerics(unweighted_loss, "loss")

    loss = tf.losses.compute_weighted_loss(unweighted_loss, weights=weights)
    loss = tf.check_numerics(loss, "loss")
  
    return loss