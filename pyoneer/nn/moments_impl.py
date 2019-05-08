from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from pyoneer.math import math_ops


def moments_from_range(minval, maxval):
    """
    Compute element-wise mean and variance from min and max values.

    Args:
        minval: A tensor of minimum values.
        maxval: A tensor of maximum values.

    Returns:
        Tuple of (mean, variance).
    """
    mean = (maxval + minval) / 2
    variance = tf.square((maxval - minval) / 2)
    return mean, variance


class StreamingMoments(tf.keras.Model):
    """
    Compute accurate moments on a batch basis.

    Args:
        shape: Shape of the moments. Used to infer the reduction axes.
        dtype: Optional dtype.
    """

    def __init__(self, shape, dtype=tf.float32, **kwargs):
        super(StreamingMoments, self).__init__(**kwargs)
        self.count = tf.Variable(tf.zeros(shape=shape, dtype=tf.int64), trainable=False)
        self.mean = tf.Variable(tf.zeros(shape=shape, dtype=dtype), trainable=False)
        self.var_sum = tf.Variable(tf.zeros(shape=shape, dtype=dtype), trainable=False)

    @property
    def variance(self):
        return tf.where(
            self.count > 1,
            self.var_sum / tf.cast(self.count - 1, self.var_sum.dtype),
            tf.zeros_like(self.var_sum),
        )

    @property
    def std(self):
        return tf.where(
            self.count > 1, tf.sqrt(self.variance), tf.zeros_like(self.var_sum)
        )

    def call(self, inputs, weights=1.0, training=None):
        """
        Update moments using a new batch of inputs.

        Args:
            inputs: Input tensor.
            weights: Optional weights.
            training: Boolean indicating whether or not to update the moments.

        Returns:
            Tuple of (mean, variance).
        """
        if training:
            inputs = tf.convert_to_tensor(inputs)

            ndims = inputs.shape.ndims - self.mean.shape.ndims
            axes = list(range(ndims))

            weight_sum = tf.reduce_sum(weights, axis=axes)
            count_delta = tf.cast(weight_sum, tf.int64)
            new_count = self.count + count_delta

            mean_delta = tf.where(
                count_delta > 1,
                (
                    tf.reduce_sum((inputs - self.mean) * weights, axis=axes)
                    / tf.cast(new_count, tf.float32)
                ),
                math_ops.safe_divide(
                    tf.reduce_sum(inputs * weights, axis=axes), weight_sum
                ),
            )
            new_mean = self.mean + mean_delta

            var_delta = tf.reduce_sum(
                (inputs - self.mean) * (inputs - new_mean) * weights, axis=axes
            )
            new_var_sum = self.var_sum + var_delta

            self.count.assign(new_count)
            self.mean.assign(new_mean)
            self.var_sum.assign(new_var_sum)

        return self.mean, self.variance


class ExponentialMovingMoments(tf.keras.Model):
    """
    Compute moments as an exponential moving average using the update rule:

    ```
    mean = (1 - rate) * old_mean + rate * new_mean
    variance = (1 - rate) * old_variance + rate * new_variance
    ```

    Args:
        shape: Shape of the moments. Used to infer the reduction axes.
        rate: Update rate in the range [0, 1] of the exponential moving
            average. Smaller values update faster.
        dtype: Optional dtype.
    """

    def __init__(self, shape, rate, dtype=tf.float32, **kwargs):
        super(ExponentialMovingMoments, self).__init__(**kwargs)
        self.rate = rate
        self.count = tf.Variable(tf.zeros(shape=shape, dtype=tf.int64), trainable=False)
        self.mean = tf.Variable(tf.zeros(shape=shape, dtype=dtype), trainable=False)
        self.variance = tf.Variable(tf.zeros(shape=shape, dtype=dtype), trainable=False)

    @property
    def std(self):
        return tf.where(
            self.count > 0, tf.sqrt(self.variance), tf.zeros_like(self.variance)
        )

    def call(self, inputs, weights=1.0, training=None):
        """
        Update moments using a new batch of inputs.

        Args:
            inputs: Input tensor.
            weights: Optional weights.
            training: Boolean indicating whether or not to update the moments.

        Returns:
            Tuple of (mean, variance).
        """
        if training:
            inputs = tf.convert_to_tensor(inputs)

            ndims = inputs.shape.ndims - self.mean.shape.ndims
            axes = list(range(ndims))

            weight_sum = tf.reduce_sum(weights, axis=axes)
            count_delta = tf.cast(weight_sum, tf.int64)
            new_count = self.count + count_delta

            mean, variance = tf.nn.weighted_moments(
                inputs, axes=axes, frequency_weights=weights
            )

            # mask values
            mean = tf.where(weight_sum > 0, mean, self.mean)
            variance = tf.where(weight_sum > 0, variance, self.variance)

            new_mean = tf.where(
                tf.logical_and(new_count > 1, weight_sum > 0),
                self.mean * self.rate + mean * (1 - self.rate),
                mean,
            )
            new_variance = tf.where(
                tf.logical_and(new_count > 1, weight_sum > 0),
                self.variance * self.rate + variance * (1 - self.rate),
                variance,
            )

            self.mean.assign(new_mean)
            self.variance.assign(new_variance)
            self.count.assign(new_count)

        return self.mean, self.variance
