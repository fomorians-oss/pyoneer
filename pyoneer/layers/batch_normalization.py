from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python import keras

class BatchNormBlock(tf.keras.layers.Wrapper):
    """Wrapper around a tf.keras layer that applies batch normalization to its
    output before applying an activation function.

    Args:
        layer: The tf.keras layer to wrap around. You'll usually want `layer`'s
            activation function to argument to be set to `None`.
        activation: The activation function to apply. Can be given as any of
            the standard ways to set an activation function on a Keras layer
            (function name as a string, callable function handle, etc.).
        batchnorm_kwargs: Keyword arguments to use when initializing the
            `BatchNormalization` layer, passed in as a `dict`.
    """
    def __init__(self, layer, activation, batchnorm_kwargs=None, **kwargs):
        super(BatchNormBlock, self).__init__(layer, **kwargs)
        if batchnorm_kwargs is None:
            batchnorm_kwargs = {}
        self.bn = tf.keras.layers.BatchNormalization(**batchnorm_kwargs)
        self.activation = keras.activations.get(activation)

    def call(self, inputs, training=False):
        x = self.layer(inputs)
        x = self.bn(x, training=training)
        return self.activation(x)
