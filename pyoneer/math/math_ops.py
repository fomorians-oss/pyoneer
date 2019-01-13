from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from pyoneer.math import logical_ops


def moments_from_range(minval, maxval):
    """
    Compute element-wise loc and scale from min and max values.

    Args:
        minval: A tensor of minimum values.
        maxval: A tensor of maximum values.

    Returns:
        Tuple of (loc, scale).
    """
    loc = (maxval + minval) / 2
    variance = tf.square((maxval - minval) / 2)
    return loc, variance


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
    return safe_divide((x - loc), scale) * weights


def denormalize(x, loc, scale, weights=1.0):
    """
    De-normalizes an input.

    Args:
        x: a possibly normalized Tensor.
        loc: expected loc.
        scale: expected scale.
        weights: optional weights.

    Returns:
        A de-normalized Tensor.
    """
    return ((x * scale) + loc) * weights
