from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from pyoneer.math import math_ops


def range_moments(minval, maxval):
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


class Moments(tf.keras.Model):
    """
    Base class for moment containers.
    """

    def normalize(self, inputs, sample_weight=1.0):
        return math_ops.normalize(
            inputs, loc=self.mean, scale=self.std, sample_weight=sample_weight
        )

    def denormalize(self, inputs, sample_weight=1.0):
        return math_ops.denormalize(
            inputs, loc=self.mean, scale=self.std, sample_weight=sample_weight
        )


class StaticMoments(Moments):
    """
    Static moments.

    Args:
        mean: Mean of moments.
        variance: Variance of moments.
    """

    def __init__(self, mean, variance, **kwargs):
        super(StaticMoments, self).__init__(**kwargs)
        self.mean = tf.Variable(mean, trainable=False)
        self.variance = tf.Variable(variance, trainable=False)

    @property
    def std(self):
        return tf.sqrt(self.variance)


class StreamingMoments(Moments):
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
    def std(self):
        return tf.where(
            self.count > 1, tf.sqrt(self.variance), tf.zeros_like(self.variance)
        )

    @property
    def variance(self):
        return tf.where(
            self.count > 1,
            self.var_sum / tf.cast(self.count - 1, self.var_sum.dtype),
            tf.zeros_like(self.var_sum),
        )

    def call(self, inputs, sample_weight=1.0, training=None):
        """
        Update moments using a new batch of inputs.

        Args:
            inputs: Input tensor.
            sample_weight: Optional sample_weight.
            training: Boolean indicating whether or not to update the moments.

        Returns:
            Tuple of (mean, variance).
        """
        if training:
            inputs = tf.convert_to_tensor(inputs)

            ndims = inputs.shape.ndims - self.mean.shape.ndims
            axes = list(range(ndims))

            weight_sum = tf.reduce_sum(sample_weight, axis=axes)
            count_delta = tf.cast(weight_sum, tf.int64)
            new_count = self.count + count_delta

            mean_delta = tf.where(
                count_delta > 1,
                (
                    tf.reduce_sum((inputs - self.mean) * sample_weight, axis=axes)
                    / tf.cast(new_count, tf.float32)
                ),
                math_ops.safe_divide(
                    tf.reduce_sum(inputs * sample_weight, axis=axes), weight_sum
                ),
            )
            new_mean = self.mean + mean_delta

            var_delta = tf.reduce_sum(
                (inputs - self.mean) * (inputs - new_mean) * sample_weight, axis=axes
            )
            new_var_sum = self.var_sum + var_delta

            self.count.assign(new_count)
            self.mean.assign(new_mean)
            self.var_sum.assign(new_var_sum)

        return self.mean, self.variance


class ExponentialMovingMoments(Moments):
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
            self.count > 1, tf.sqrt(self.variance), tf.zeros_like(self.variance)
        )

    def call(self, inputs, sample_weight=1.0, training=None):
        """
        Update moments using a new batch of inputs.

        Args:
            inputs: Input tensor.
            sample_weight: Optional sample_weight.
            training: Boolean indicating whether or not to update the moments.

        Returns:
            Tuple of (mean, variance).
        """
        if training:
            inputs = tf.convert_to_tensor(inputs)

            ndims = inputs.shape.ndims - self.mean.shape.ndims
            axes = list(range(ndims))

            weight_sum = tf.reduce_sum(sample_weight, axis=axes)
            count = tf.cast(weight_sum, tf.int64)
            new_count = self.count + count

            mean, variance = tf.nn.weighted_moments(
                inputs, axes=axes, frequency_weights=sample_weight
            )

            moving_mean = self.mean * self.rate + mean * (1 - self.rate)
            moving_variance = self.variance * self.rate + variance * (1 - self.rate)

            new_mean = tf.where(self.count > 0, moving_mean, mean)
            new_variance = tf.where(self.count > 0, moving_variance, variance)

            new_mean = tf.where(count > 0, new_mean, self.mean)
            new_variance = tf.where(count > 0, new_variance, self.variance)

            self.mean.assign(new_mean)
            self.variance.assign(new_variance)
            self.count.assign(new_count)

        return self.mean, self.variance
