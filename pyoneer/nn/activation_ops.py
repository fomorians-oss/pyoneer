from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def swish(x):
    return x * tf.nn.sigmoid(x)