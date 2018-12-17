from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def swish(x):
    """
    Compute the Swish, self-gating, activation function: `x * sigmoid(x)`.

    Args:
        x: Tensor

    Returns:
        Tensor of same dimension as `x`.
    """
    y = x * tf.sigmoid(x)
    y = tf.check_numerics(y, 'swish')
    return y
