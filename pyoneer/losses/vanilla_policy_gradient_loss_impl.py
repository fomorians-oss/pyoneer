from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from trfl import policy_gradient_ops


def vanilla_policy_gradient_loss(log_probs,
                                 advantages,
                                 weights=1.0):
    """Computes the Vanilla policy gradient loss.
    
    Args:
        log_probs: Log probabilities of taking actions under a policy.
        advantages: Advantage function estimation.
        weights: Tensor of shape `[B, T]` containing weights (1. or 0.).

    Returns:
        A scalar tensor 
    """

    advantages = tf.stop_gradient(advantages)

    losses = -log_probs * advantages
    losses = tf.check_numerics(losses, "loss")

    loss = tf.losses.compute_weighted_loss(losses, weights=weights)
    loss = tf.check_numerics(loss, "losses")
  
    return loss

def entropy_loss(entropy, weights=1.):
    """Computes the entropy loss.

    Use this to encourage the policy posterior distribution to expand or collapse during training.

    Args: 
        entropy: A Tensor of entropies.
        weights: Tensor of shape `[B, T]` containing weights.

    Returns: Tensor of shape []
    """
    entropy_loss = -tf.losses.compute_weighted_loss(
        losses=entropy,
        weights=weights)
    entropy_loss = tf.check_numerics(entropy_loss, "entropy_loss")
    return entropy_loss
