from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def vanilla_policy_gradient_loss(log_probs,
                                 advantages,
                                 weights=1.0):
    """ Computes the Vanilla PG loss.
    
    Args:
      log_probs: Log probabilities of taking actions under a policy.
      advantages: Advantage function estimation.
      weights: Optional tensor for weighting the losses.
    Returns tensor 
    """

    advantages = tf.stop_gradient(advantages)
    weights = tf.convert_to_tensor(weights)

    losses = -log_probs * advantages
    losses = tf.check_numerics(losses, "loss")

    loss = tf.losses.compute_weighted_loss(losses, weights=weights)
    loss = tf.check_numerics(loss, "losses")
  
    return loss
