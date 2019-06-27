from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_probability as tfp


class SoftplusInverse(tf.keras.initializers.Initializer):
    """
    Initializer that generates tensors initialized to `log(exp(scale) - 1)`.

    Args:
        scale: scale of the initializer output.
        dtype: dtype of the operation.
    """

    def __init__(self, scale=1.0):
        self.scale = scale

    def __call__(self, shape, dtype=tf.float32):
        return tf.constant(
            tfp.math.softplus_inverse(self.scale), dtype=dtype, shape=shape
        )

    def get_config(self):
        return {"scale": self.scale, "dtype": self.dtype.name}
