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
    weights = tf.convert_to_tensor(weights)

    losses = -log_probs * advantages
    losses = tf.check_numerics(losses, "loss")

    loss = tf.losses.compute_weighted_loss(losses, weights=weights)
    loss = tf.check_numerics(loss, "losses")
  
    return loss

def vanilla_policy_entropy_loss(policy, 
                                entropy_scale,
                                trainable_variables=None,
                                weights=1.):
    """Compute the entropy loss. 
    
    Args:
        policy: a subclass of tf.keras.Model 
        entropy_scale: scalar or Tensor of shape `[B, T]` containing the entropy loss scale.
        weights: Tensor of shape `[B, T]` containing weights (1. or 0.).

    Returns:
        The entropy loss Tensor of shape [] 

    """

    losses = policy_gradient_ops.policy_entropy_loss(
        policy,
        trainable_variables,
        lambda policies: entropy_scale).loss
    losses = tf.check_numerics(losses, "losses")

    return tf.losses.compute_weighted_loss(losses, weights=weights)
