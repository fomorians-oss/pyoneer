from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf

RADIANS_TO_DEGREES = 180 / math.pi
DEGREES_TO_RADIANS = math.pi / 180


def to_radians(x):
    """
    Convert the inputs from degrees to radians.
    """
    return x * DEGREES_TO_RADIANS


def to_degrees(x):
    """
    Convert the inputs from radians to degrees.
    """
    return x * RADIANS_TO_DEGREES


def to_cartesian(phi, rho=1):
    """
    Convert the inputs from polar to cartesian coordinates.
    """
    x = rho * tf.cos(phi)
    y = rho * tf.sin(phi)
    return x, y


def to_polar(x, y):
    """
    Convert the inputs from cartesian to polar coordinates.
    """
    rho = tf.sqrt(tf.square(x) + tf.square(y))
    phi = tf.atan2(y, x)
    return rho, phi
