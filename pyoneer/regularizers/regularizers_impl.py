from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class DropoutL2(tf.keras.regularizers.Regularizer):
    """
    Regularize a kernel scaled by the inverse keep probability of a
    dropout layer.

    Args:
        dropout: Dropout instance with a `rate` property.
        scale: Scalar coefficient for the regularization.
    """

    def __init__(self, dropout, scale=1e-6):
        self.dropout = dropout
        self.scale = scale

    def __call__(self, x):
        regularization = (
            self.scale
            * tf.math.reduce_sum(tf.math.square(x))
            / (1.0 - self.dropout.rate)
        )
        return regularization

    def get_config(self):
        return {"scale": self.scale}
