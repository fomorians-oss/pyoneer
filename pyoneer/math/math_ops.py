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
