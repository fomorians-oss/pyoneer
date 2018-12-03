from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def isclose(a, b, rtol=1e-5, atol=1e-8):
    """
    Returns a boolean tensor where two arrays are element-wise equal within a
    tolerance.

    The relative difference (rtol * abs(b)) and the absolute difference atol
    are added together to compare against the absolute difference between a
    and b.
    """
    a = tf.convert_to_tensor(a)
    b = tf.convert_to_tensor(b)
    rtol = tf.convert_to_tensor(rtol)
    atol = tf.convert_to_tensor(atol)
    return tf.abs(a - b) <= (atol + rtol * tf.abs(b))
