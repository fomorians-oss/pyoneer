from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from pyoneer.math import logical_ops


def safe_divide(x, y, rtol=1e-5, atol=1e-8):
    """
    Safely divide x by y while avoiding dividing by zero.
    """
    y = tf.where(
        logical_ops.isclose(y, 0.0, rtol=rtol, atol=atol), tf.ones_like(y), y)
    return x / y


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
    return x


def normalize(x, loc, scale, weights=1.0):
    """
    Normalizes an input.

    Args:
        x: a possibly un-normalized Tensor.
        loc: expected loc.
        scale: expected scale.
        weights: optional weights.

    Returns:
        A normalized Tensor.
    """
    x = tf.convert_to_tensor(x)
    loc = tf.convert_to_tensor(loc)
    scale = tf.convert_to_tensor(scale)
    weights = tf.convert_to_tensor(weights)
    return safe_divide((x - loc), scale) * weights


def denormalize(x, loc, scale, weights=1.0):
    """
    De-normalizes an input.

    Args:
        x: A tensor to denormalize.
        loc: A loc tensor.
        scale: A scale tensor.
        weights: Optional weights tensor.

    Returns:
        A de-normalized Tensor.
    """
    x = tf.convert_to_tensor(x)
    loc = tf.convert_to_tensor(loc)
    scale = tf.convert_to_tensor(scale)
    weights = tf.convert_to_tensor(weights)
    return ((x * scale) + loc) * weights
