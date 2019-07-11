from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

import six
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


@six.add_metaclass(abc.ABCMeta)
class Moments(tf.keras.layers.Layer):
    """
    Base class for moment containers.
    """

    @property
    @abc.abstractmethod
    def mean(self):
        raise NotImplementedError("Must be implemented in subclasses.")

    @property
    @abc.abstractmethod
    def variance(self):
        raise NotImplementedError("Must be implemented in subclasses.")

    @property
    def std(self):
        return tf.sqrt(self.variance)

    @abc.abstractmethod
    def update_state(self, *args, **kwargs):
        raise NotImplementedError("Must be implemented in subclasses.")

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
        self._mean = tf.Variable(mean, trainable=False)
        self._variance = tf.Variable(variance, trainable=False)

    @property
    def mean(self):
        return self._mean

    @property
    def variance(self):
        return self._variance

    def update_state(self, *args, **kwargs):
        raise NotImplementedError("StaticMoments cannot be updated.")


class StreamingMoments(Moments):
    """
    Compute accurate moments on a batch basis.

    Args:
        shape: Shape of the moments. Used to infer the reduction axes.
        dtype: Optional dtype.
    """

    def __init__(self, shape, dtype=tf.float32, **kwargs):
        super(StreamingMoments, self).__init__(**kwargs)
        self._count = self.add_weight(
            name="count",
            initializer="zeros",
            shape=shape,
            dtype=tf.int64,
            trainable=False,
        )
        self._mean = self.add_weight(
            name="mean", initializer="zeros", shape=shape, dtype=dtype, trainable=False
        )
        self._var_sum = self.add_weight(
            name="var_sum",
            initializer="zeros",
            shape=shape,
            dtype=dtype,
            trainable=False,
        )

    @property
    def count(self):
        return self._count

    @property
    def mean(self):
        return self._mean

    @property
    def variance(self):
        return tf.where(
            self._count > 1,
            self._var_sum / tf.cast(self._count - 1, self._var_sum.dtype),
            tf.zeros_like(self._var_sum),
        )

    def update_state(self, inputs, sample_weight=None):
        """
        Update moments using a new batch of inputs.

        Args:
            inputs: Input tensor.
            sample_weight: Optional sample weight.
        """
        inputs = tf.convert_to_tensor(inputs)

        if sample_weight is None:
            sample_weight = 1.0

        ndims = inputs.shape.ndims - self._mean.shape.ndims
        axes = list(range(ndims))

        weight_sum = tf.reduce_sum(sample_weight, axis=axes)
        count_delta = tf.cast(weight_sum, tf.int64)
        new_count = self._count + count_delta

        mean_delta = tf.where(
            count_delta > 1,
            (
                tf.reduce_sum((inputs - self._mean) * sample_weight, axis=axes)
                / tf.cast(new_count, tf.float32)
            ),
            math_ops.safe_divide(
                tf.reduce_sum(inputs * sample_weight, axis=axes), weight_sum
            ),
        )
        new_mean = self._mean + mean_delta

        var_delta = tf.reduce_sum(
            (inputs - self._mean) * (inputs - new_mean) * sample_weight, axis=axes
        )
        new_var_sum = self._var_sum + var_delta

        self._count.assign(new_count)
        self._mean.assign(new_mean)
        self._var_sum.assign(new_var_sum)


class ExponentialMovingMoments(Moments):
    """
    Compute moments as an exponential moving average using the update rule:

    ```
    mean = (1 - rate) * old_mean + rate * new_mean
    variance = (1 - rate) * old_variance + rate * new_variance
    ```

    Args:
        rate: Update rate in the range [0, 1] of the exponential moving
            average. Smaller values update faster.
        shape: Shape of the moments. Used to infer the reduction axes.
        dtype: Optional dtype.
    """

    def __init__(self, rate, shape, dtype=tf.float32, **kwargs):
        super(ExponentialMovingMoments, self).__init__(**kwargs)
        self._count = self.add_weight(
            name="count",
            initializer="zeros",
            shape=shape,
            dtype=tf.int64,
            trainable=False,
        )
        self._mean = self.add_weight(
            name="mean", initializer="zeros", shape=shape, dtype=dtype, trainable=False
        )
        self._variance = self.add_weight(
            name="variance",
            initializer="zeros",
            shape=shape,
            dtype=dtype,
            trainable=False,
        )
        self.rate = rate

    @property
    def count(self):
        return self._count

    @property
    def mean(self):
        return self._mean

    @property
    def variance(self):
        return self._variance

    def update_state(self, inputs, sample_weight=None):
        """
        Update moments using a new batch of inputs.

        Args:
            inputs: Input tensor.
            sample_weight: Optional sample weight.
        """
        inputs = tf.convert_to_tensor(inputs)

        if sample_weight is None:
            sample_weight = 1.0

        ndims = inputs.shape.ndims - self._mean.shape.ndims
        axes = list(range(ndims))

        weight_sum = tf.reduce_sum(sample_weight, axis=axes)
        count = tf.cast(weight_sum, tf.int64)
        new_count = self._count + count

        mean, variance = tf.nn.weighted_moments(
            inputs, axes=axes, frequency_weights=sample_weight
        )

        moving_mean = self._mean * self.rate + mean * (1 - self.rate)
        moving_variance = self._variance * self.rate + variance * (1 - self.rate)

        new_mean = tf.where(self._count > 0, moving_mean, mean)
        new_variance = tf.where(self._count > 0, moving_variance, variance)

        new_mean = tf.where(count > 0, new_mean, self._mean)
        new_variance = tf.where(count > 0, new_variance, self._variance)

        self._count.assign(new_count)
        self._mean.assign(new_mean)
        self._variance.assign(new_variance)
