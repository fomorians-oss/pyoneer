import tensorflow as tf

import numpy as np


def normalize(x, low, high):
    """
    Normalize to standard distribution.
    """
    x = tf.convert_to_tensor(x)
    mean = (high + low) / 2
    stddev = (high - low) / 2
    stddev = np.where(np.isclose(stddev, 0.0), 1.0, stddev)
    x = (x - mean) / stddev
    return x


def denormalize(x, low, high):
    """
    Denormalize to original distribution.
    """
    x = tf.convert_to_tensor(x)
    mean = (high + low) / 2
    stddev = (high - low) / 2
    stddev = np.where(np.isclose(stddev, 0.0), 1.0, stddev)
    x = (x * stddev) + mean
    return x


class HighLowNormalizer(tf.keras.Model):
    """
    Normalize and denormalize according to high and low.
    """
    def __init__(self, low, high):
        super(HighLowNormalizer, self).__init__()
        self.low = tf.convert_to_tensor(low)
        self.high = tf.convert_to_tensor(high)
        self.output_size = self.low.shape[-1]

    def call(self, inputs, weights=1.):
        return normalize(inputs, self.low, self.high) * weights

    def inverse(self, inputs, weights=1.):
        return denormalize(inputs, self.low, self.high) * weights