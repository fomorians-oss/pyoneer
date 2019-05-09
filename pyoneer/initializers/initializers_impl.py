from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_probability as tfp


class SoftplusInverse(tf.initializers.Initializer):
    """
    Initializer that generates tensors initialized to `log(exp(scale) - 1)`.

    Args:
        scale: scale of the initializer output.
        dtype: dtype of the operation.
    """

    def __init__(self, scale=1.0, dtype=tf.float32):
        self.scale = tf.convert_to_tensor(scale)
        self.dtype = tf.dtypes.as_dtype(dtype)

    def __call__(self, shape, dtype=None, verify_shape=None):
        if dtype is None:
            dtype = self.dtype

        return tf.constant(
            tfp.distributions.softplus_inverse(self.scale),
            dtype=dtype,
            shape=shape,
            verify_shape=verify_shape,
        )

    def get_config(self):
        return {"scale": self.scale, "dtype": self.dtype.name}
