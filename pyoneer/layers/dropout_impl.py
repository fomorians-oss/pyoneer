from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class ConcreteDropout(tf.keras.layers.Wrapper):
    """
    Implementation of concrete dropout (Gal et al. 2017).

    Args:
        layer: Apply dropout to the inputs of `layer` when called.
        initial_rate: Initial dropout rate to apply.

    Usage:

        base_layer = tf.keras.layers.Dense(units=3)
        dropout_layer = ConcreteDropout(base_layer)
        dropout_layer(inputs)

    References:
        Concrete Dropout. Gal et al. 2017. https://arxiv.org/abs/1705.07832
    """

    def __init__(
        self,
        layer,
        initial_rate=0.1,
        kernel_regularization=1e-6,
        dropout_regularization=1e-5,
        noise_shape=None,
        seed=None,
        **kwargs
    ):
        super(ConcreteDropout, self).__init__(layer, **kwargs)
        self.initial_rate = initial_rate
        self.kernel_regularization = kernel_regularization
        self.dropout_regularization = dropout_regularization
        self.noise_shape = noise_shape
        self.seed = seed
        self.supports_masking = True
        self.rate_logit = None

    def build(self, input_shape):
        super(ConcreteDropout, self).build(input_shape)

        initial_rate_norm = np.log(self.initial_rate) - np.log(1.0 - self.initial_rate)
        rate_initializer = tf.keras.initializers.Constant(initial_rate_norm)

        self.rate_logit = self.add_weight(
            name="rate_logit", shape=(), initializer=rate_initializer, trainable=True
        )

        if self.noise_shape is None:
            self.noise_shape = input_shape

        input_dim = np.prod(self.noise_shape[1:])

        kernel_regularizer = (
            self.kernel_regularization
            * tf.math.reduce_sum(tf.math.square(self.layer.kernel))
            / (1.0 - self.rate)
        )

        dropout_regularizer = self.rate * tf.math.log(self.rate)
        dropout_regularizer += (1.0 - self.rate) * tf.math.log(1.0 - self.rate)
        dropout_regularizer *= self.dropout_regularization * input_dim

        regularizer = tf.math.reduce_sum(kernel_regularizer + dropout_regularizer)
        self.add_loss(regularizer)

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    @property
    def rate(self):
        assert (
            self.rate_logit is not None
        ), "the layer must be called before accessing rate"
        return tf.math.sigmoid(self.rate_logit)

    def concrete_dropout(self, inputs):
        epsilon = tf.keras.backend.cast_to_floatx(tf.keras.backend.epsilon())
        temperature = 0.1

        unif_noise = tf.random.uniform(shape=self.noise_shape, seed=self.seed)
        drop_prob = (
            tf.math.log(self.rate + epsilon)
            - tf.math.log(1.0 - self.rate + epsilon)
            + tf.math.log(unif_noise + epsilon)
            - tf.math.log(1.0 - unif_noise + epsilon)
        )
        drop_prob = tf.math.sigmoid(drop_prob / temperature)
        random_tensor = 1.0 - drop_prob

        retain_prob = 1.0 - self.rate
        inputs *= random_tensor
        inputs /= retain_prob
        return inputs

    def call(self, inputs, training=None):
        return self.layer.call(self.concrete_dropout(inputs))

    def get_config(self):
        config = {
            "initial_rate": self.initial_rate,
            "kernel_regularization": self.kernel_regularization,
            "dropout_regularization": self.dropout_regularization,
            "noise_shape": self.noise_shape,
            "seed": self.seed,
        }
        base_config = super(Dropout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
