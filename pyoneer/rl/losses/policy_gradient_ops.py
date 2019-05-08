from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.keras.utils import losses_utils


def policy_gradient_loss(log_probs, advantages, sample_weight=1.0):
    """
    Computes the Vanilla policy gradient loss.

    Args:
        log_probs: Log probabilities of taking actions under a policy.
        advantages: Advantage estimation.
        sample_weight: Tensor of shape `[B, T]` containing sample_weight (1. or 0.).

    Returns:
        A scalar tensor.
    """
    advantages = tf.stop_gradient(advantages)

    losses = -log_probs * advantages
    losses = tf.debugging.check_numerics(losses, "loss")

    loss = losses_utils.compute_weighted_loss(losses, sample_weight=sample_weight)
    loss = tf.debugging.check_numerics(loss, "losses")

    return loss


def clipped_policy_gradient_loss(
    log_probs, log_probs_anchor, advantages, epsilon_clipping=0.2, sample_weight=1.0
):
    """
    Computes the clipped surrogate objective found in
    [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347) based on
    clipped probability ratios.

    Args:
        log_probs: Log probabilities of taking actions under a policy.
        log_probs_anchor: Log probabilities of taking actions under an anchor
            policy which is updated less frequently.
        advantages: Advantage estimation.
        epsilon_clipping: Scalar for clipping the policy ratio.
        sample_weight: Optional tensor for weighting the losses.

    Returns:
        A scalar tensor.
    """
    log_probs_anchor = tf.stop_gradient(log_probs_anchor)
    advantages = tf.stop_gradient(advantages)

    ratio = tf.exp(log_probs - log_probs_anchor)
    ratio = tf.debugging.check_numerics(ratio, "ratio")

    surrogate1 = ratio * advantages
    surrogate1 = tf.debugging.check_numerics(surrogate1, "surrogate1")

    surrogate2 = (
        tf.clip_by_value(ratio, 1 - epsilon_clipping, 1 + epsilon_clipping) * advantages
    )
    surrogate2 = tf.debugging.check_numerics(surrogate2, "surrogate2")

    surrogate_min = tf.minimum(surrogate1, surrogate2)
    surrogate_min = tf.debugging.check_numerics(surrogate_min, "surrogate_min")

    loss = -losses_utils.compute_weighted_loss(
        losses=surrogate_min, sample_weight=sample_weight
    )
    loss = tf.debugging.check_numerics(loss, "loss")
    return loss
