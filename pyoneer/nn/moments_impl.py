from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.eager as tfe

from pyoneer.math import math_ops


class StreamingMoments(tf.keras.Model):
    def __init__(self, shape, dtype=tf.float32, **kwargs):
        super(StreamingMoments, self).__init__(**kwargs)
        self.count = tfe.Variable(
            tf.zeros(shape=shape, dtype=tf.int64), trainable=False)
        self.mean = tfe.Variable(
            tf.zeros(shape=shape, dtype=dtype), trainable=False)
        self.var_sum = tfe.Variable(
            tf.zeros(shape=shape, dtype=dtype), trainable=False)

    @property
    def variance(self):
        return tf.where(
            self.count > 1,
            self.var_sum / tf.cast(self.count - 1, self.var_sum.dtype),
            tf.zeros_like(self.var_sum))

    @property
    def std(self):
        return tf.where(self.count > 1, tf.sqrt(self.variance),
                        tf.zeros_like(self.var_sum))

    def call(self, inputs, weights=1.0, training=None):
        if training:
            inputs = tf.convert_to_tensor(inputs)

            ndims = inputs.shape.ndims - self.mean.shape.ndims
            axis = list(range(ndims))

            weight_sum = tf.reduce_sum(weights, axis=axis)
            count_delta = tf.to_int64(weight_sum)
            new_count = self.count + count_delta

            mean_delta = tf.where(
                count_delta > 1, (tf.reduce_sum(
                    (inputs - self.mean) * weights, axis=axis) /
                                  tf.to_float(new_count)),
                math_ops.safe_divide(
                    tf.reduce_sum(inputs * weights, axis=axis), weight_sum))
            new_mean = self.mean + mean_delta

            var_delta = tf.reduce_sum(
                (inputs - self.mean) * (inputs - new_mean) * weights,
                axis=axis)
            new_var_sum = self.var_sum + var_delta

            import ipdb
            ipdb.set_trace()

            self.count.assign(new_count)
            self.mean.assign(new_mean)
            self.var_sum.assign(new_var_sum)

        return self.mean, self.variance
