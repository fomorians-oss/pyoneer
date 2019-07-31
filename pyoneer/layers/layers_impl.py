from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from pyoneer.activations import activations_impl
from pyoneer.math import angle_ops, math_ops


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
    """

    def __init__(self, depth, **kwargs):
        super(OneHotEncoder, self).__init__(**kwargs)
        self.depth = depth

    def call(self, inputs):
        """
        Encode the inputs.

        Args:
            inputs: An integer tensor.

        Returns:
            The one-hot encoded inputs.
        """
        inputs = tf.cast(inputs, tf.int64)
        outputs = tf.one_hot(inputs, self.depth)
        outputs = tf.debugging.check_numerics(outputs, "outputs")
        return outputs

    def get_config(self):
        config = {"depth": self.depth}
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


class Nest(tf.keras.layers.Layer):
    """
    Maps an arbitrarily nested structure of layers to an equivalent
    structure of input features using the `tf.nest` API.

    Example:

    ```
    nest_layer = Nest({
        "category": OneHotEncoder(depth=3),
        "angle": AngleEncoder(degrees=True),
        "nested": {
            "category": OneHotEncoder(depth=3),
            "angle": AngleEncoder(degrees=True),
        }
    })
    features = {
        "category": tf.constant([1]),
        "angle": tf.constant([[-45.0]]),
        "nested": {
            "category": tf.constant([2]),
            "angle": tf.constant([[+45.0]]),
        }
    }
    outputs = nest_layer(features)  # shape: [1, (2 + 3) * 2]
    ```

    Args:
        layers: An arbitrarily nested structure of layers.
    """

    def __init__(self, layers, **kwargs):
        super(Nest, self).__init__(**kwargs)
        self.layers = layers

    def call(self, inputs):
        tf.nest.assert_same_structure(self.layers, inputs)

        def call_layer(layer, inputs):
            return layer(inputs)

        inputs_mapped = tf.nest.map_structure(call_layer, self.layers, inputs)
        inputs_flattened = tf.nest.flatten(inputs_mapped)
        outputs = tf.concat(inputs_flattened, axis=-1)
        return outputs


class ConcreteDropout(tf.keras.layers.Layer):
    """
    Implementation of concrete dropout.

    Args:
        initial_rate: Initial dropout rate to apply.

    Usage:
        dropout_layer = pynr.layers.ConcreteDropout()
        scale = 1e-5 * effective_input_dim
        dropout_regularizer = pyrl.regularizers.ConcreteDropout(dropout_layer, scale)
        dropout_layer.add_loss(dropout_regularizer(dropout_layer.rate_logit))
        kernel_regularizer = pynr.regularizers.DropoutL2(dropout_layer)

        base_layer = tf.keras.layers.Dense(
            units=3,
            kernel_regularizer=kernel_regularizer)

        inputs = dropout_layer(inputs)
        hidden = base_layer(inputs)

    References:
        Concrete Dropout. Gal et al. 2017. https://arxiv.org/abs/1705.07832
    """

    def __init__(
        self, initial_rate=0.1, temperature=0.1, noise_shape=None, seed=None, **kwargs
    ):
        super(ConcreteDropout, self).__init__(**kwargs)
        self.initial_rate = initial_rate
        self.noise_shape = noise_shape
        self.seed = seed
        self.supports_masking = True
        self.rate_logit = None
        self.temperature = temperature

        initial_rate_norm = math_ops.sigmoid_inverse(self.initial_rate)
        self.rate_logit = tf.Variable(initial_value=initial_rate_norm, trainable=True)

    @property
    def rate(self):
        return tf.math.sigmoid(self.rate_logit)

    def call(self, inputs, training=None):
        epsilon = tf.keras.backend.cast_to_floatx(tf.keras.backend.epsilon())

        noise_shape = self.noise_shape
        if noise_shape is None:
            noise_shape = inputs.shape
        noise = tf.random.uniform(
            shape=noise_shape, minval=epsilon, maxval=1.0, seed=self.seed
        )
        rate = self.rate + epsilon

        drop_prob = (
            tf.math.log(rate)
            - tf.math.log(1.0 - rate)
            + tf.math.log(noise)
            - tf.math.log(1.0 - noise)
        )
        drop_prob_temp = tf.math.sigmoid(drop_prob / self.temperature)

        keep_prob_rand = 1.0 - drop_prob_temp
        keep_prob = 1.0 - rate

        outputs = (inputs * keep_prob_rand) / keep_prob
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            "initial_rate": self.initial_rate,
            "temperature": self.temperature,
            "noise_shape": self.noise_shape,
            "seed": self.seed,
        }
        base_config = super(ConcreteDropout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
