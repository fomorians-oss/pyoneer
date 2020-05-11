from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def compute_weighted_loss(losses, sample_weight=1.0):
    """
    Compute weighted loss from losses and sample weights.

    Args:
        losses: Tensor of losses.
        sample_weight: Tensor of sample weight(s).

    Returns:
        Weighted loss.
    """
    losses = losses * sample_weight
    total_loss = tf.reduce_sum(losses * sample_weight)
    num_present = tf.reduce_sum(tf.ones_like(losses) * sample_weight)
    return tf.math.divide_no_nan(total_loss, num_present)
