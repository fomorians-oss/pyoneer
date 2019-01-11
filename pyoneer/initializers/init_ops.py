from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_probability as tfp


class SoftplusInverse(tf.keras.initializers.Initializer):
    """
    Initializer that generates tensors initialized to `log(exp(value) - 1)`.
    """

    def __init__(self, value=1.0, dtype=tf.float32):
        """
        Creates a new InverseSoftplusScale initializer.

        Args:
            value: value of the initializer output.
            dtype: dtype of the operation.
        """
        self.value = tf.convert_to_tensor(value)
        self.dtype = tf.dtypes.as_dtype(dtype)

    def __call__(self,
                 shape,
                 dtype=None,
                 partition_info=None,
                 verify_shape=None):
        if dtype is None:
            dtype = self.dtype

        return tf.constant(
            tfp.distributions.softplus_inverse(self.value),
            dtype=dtype,
            shape=shape,
            verify_shape=verify_shape)

    def get_config(self):
        return {'value': self.value, 'dtype': self.dtype.name}
