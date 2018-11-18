import tensorflow as tf


def weighted_normalize(x, weights, axes=[0, 1]):
    x_mean, x_variance = tf.nn.weighted_moments(
        x, axes=axes, frequency_weights=weights, keep_dims=True)
    x_stddev = tf.sqrt(x_variance + 1e-6) + 1e-8
    return tf.where(
        tf.equal(weights, 1.0),
        (x - x_mean) / x_stddev, x)
