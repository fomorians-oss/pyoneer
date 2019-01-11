from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def layer_norm(inputs, beta, gamma):
    """Compute the layer normalization operation.

    Args:
        inputs: Tensor no less than 2-D.
        beta: Tensor size 1-D
        gamma: Tensor size 1-D

    Returns:
        A layer normalized Tensor the same dimension as `inputs`.
    """
    inputs = tf.convert_to_tensor(inputs)
    inputs_shape = inputs.shape
    inputs_rank = inputs_shape.ndims

    # Calculate the moments on the last axis (layer activations).
    norm_axes = list(range(1, inputs_rank))
    mean, variance = tf.nn.moments(inputs, norm_axes, keep_dims=True)

    # Compute layer normalization using the batch_normalization function.
    variance_epsilon = 1e-12
    outputs = tf.nn.batch_normalization(
        inputs,
        mean,
        variance,
        offset=beta,
        scale=gamma,
        variance_epsilon=variance_epsilon)
    outputs.set_shape(inputs_shape)
    return outputs
