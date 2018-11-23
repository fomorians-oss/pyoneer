import tensorflow as tf


def swish(x):
    return x * tf.nn.sigmoid(x)