from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def batched_index(values, indices):
    """
    Performs indexing on batches and sequence-batches by reducing over
    zero-masked values.

    Equivalent to `values[:, indices]`.

    Args:
        values: tensor of shape `[B, num_values]` or `[T, B, num_values]`
        indices: tensor of shape `[B]` or `[T, B]` containing indices.
    Returns:
        Tensor of shape `[B]` or `[T, B]` containing values for the given
        indices.
    """
    values = tf.convert_to_tensor(values)
    indices = tf.convert_to_tensor(indices)
    mask = tf.one_hot(indices, values.shape[-1], dtype=values.dtype)
    return tf.reduce_sum(values * mask, axis=-1)
