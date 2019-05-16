from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from pyoneer.math import logical_ops


def safe_divide(x, y, rtol=1e-5, atol=1e-8):
    """
    Safely divide x by y while avoiding dividing by zero.
    """
    y = tf.where(logical_ops.isclose(y, 0.0, rtol=rtol, atol=atol), tf.ones_like(y), y)
    return tf.debugging.check_numerics(x / y, "safe_divide")


def rescale(x, oldmin, oldmax, newmin, newmax):
    """
    Rescale from [oldmin..oldmax] to [newmin..newmax].
    """
    x = tf.convert_to_tensor(x)
    oldmin = tf.convert_to_tensor(oldmin)
    oldmax = tf.convert_to_tensor(oldmax)
    newmin = tf.convert_to_tensor(newmin)
    newmax = tf.convert_to_tensor(newmax)
    x = (x - oldmin) / (oldmax - oldmin)
    x = (x * (newmax - newmin)) + newmin
    x = tf.debugging.check_numerics(x, "rescale")
    return x


def normalize(x, loc, scale, sample_weight=1.0):
    """
    Normalizes an input.

    Args:
        x: a possibly un-normalized Tensor.
        loc: expected loc.
        scale: expected scale.
        sample_weight: optional sample_weight.

    Returns:
        A normalized Tensor.
    """
    x = tf.convert_to_tensor(x)
    loc = tf.convert_to_tensor(loc)
    scale = tf.convert_to_tensor(scale)
    sample_weight = tf.convert_to_tensor(sample_weight)
    outputs = safe_divide((x - loc), scale) * sample_weight
    outputs = tf.debugging.check_numerics(outputs, "normalize")
    return outputs


def denormalize(x, loc, scale, sample_weight=1.0):
    """
    De-normalizes an input.

    Args:
        x: A tensor to denormalize.
        loc: A loc tensor.
        scale: A scale tensor.
        sample_weight: Optional sample_weight tensor.

    Returns:
        A de-normalized Tensor.
    """
    x = tf.convert_to_tensor(x)
    loc = tf.convert_to_tensor(loc)
    scale = tf.convert_to_tensor(scale)
    sample_weight = tf.convert_to_tensor(sample_weight)
    outputs = ((x * scale) + loc) * sample_weight
    outputs = tf.debugging.check_numerics(outputs, "denormalize")
    return outputs
