from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from pyoneer.activations import activations_impl
from pyoneer.math import angle_ops


class Swish(tf.keras.layers.Layer):
    """
    Compute Swish, self-gating, activation function layer: `x * sigmoid(x)`.
    """

    def call(self, inputs):
        """
        Compute activations for the inputs.

        Args:
            inputs: A tensor.

        Returns:
            The activations.
        """
        outputs = activations_impl.swish(inputs)
        return outputs


class OneHotEncoder(tf.keras.layers.Layer):
    """
    One-hot encoding layer. Encodes the integer inputs as one-hot vectors.

    Args:
        depth: The depth of the one-hot encoding.
        dtype: The output dtype.
    """

    def __init__(self, depth, dtype=None, **kwargs):
        super(OneHotEncoder, self).__init__(**kwargs)
        self.depth = depth
        if dtype is None:
            dtype = tf.dtypes.float32
        self.output_dtype = dtype

    def call(self, inputs):
        """
        Encode the inputs.

        Args:
            inputs: An integer tensor.

        Returns:
            The one-hot encoded inputs.
        """
        inputs = tf.cast(inputs, tf.int64)
        outputs = tf.one_hot(inputs, self.depth, dtype=self.output_dtype)
        outputs = tf.debugging.check_numerics(outputs, "outputs")
        return outputs

    def get_config(self):
        config = {"depth": self.depth, "output_dtype": self.output_dtype}
        base_config = super(OneHotEncoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class AngleEncoder(tf.keras.layers.Layer):
    """
    Angle encoding layer. Encodes an angle as the cosine and sine of radians
    or degrees which will be converted to radians.

    Args:
        degrees (default: False):
            Whether the inputs are in degrees (True) or radians (False).
    """

    def __init__(self, degrees=False, **kwargs):
        super(AngleEncoder, self).__init__(**kwargs)
        self.degrees = degrees

    def call(self, inputs):
        if self.degrees:
            inputs = angle_ops.to_radians(inputs)
        x, y = angle_ops.to_cartesian(inputs)
        outputs = tf.concat([x, y], axis=-1)
        return outputs

    def get_config(self):
        config = {"degrees": self.degrees}
        base_config = super(AngleEncoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class BatchApply(tf.keras.layers.Layer):

    def __init__(self, module, num_dims=2):
        super().__init__()
        self.module = module
        self.num_dims = num_dims

    @tf.function
    def call(self, inputs, training=False):
        input_shape = tf.shape(inputs)
        input_batch_dims = input_shape[:self.num_dims]
        input_batch_size = tf.reduce_prod(input_batch_dims)

        inputs_distributed = tf.reshape(
            inputs,
            tf.concat([[input_batch_size], input_shape[self.num_dims:]],
                      axis=0))
        outputs_distributed = self.module(inputs_distributed,
                                          training=training)

        outputs_distributed_shape = tf.shape(outputs_distributed)
        outputs = tf.reshape(
            outputs_distributed,
            tf.concat([input_batch_dims, outputs_distributed_shape[1:]],
                      axis=0))

        return outputs
