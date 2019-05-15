from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from collections import OrderedDict

from pyoneer.math import angle_ops
from pyoneer.math import math_ops


class Normalizer(tf.keras.layers.Layer):
    """
    Input normalization layer. Normalizes the inputs by subtracting a mean and
    dividing by a standard deviation.

    Args:
        loc: The mean to use for normalization.
        scale: The standard deviation to use for normalization.
    """

    def __init__(self, loc, scale, **kwargs):
        super(Normalizer, self).__init__(**kwargs)
        self.loc = tf.Variable(loc, trainable=False)
        self.scale = tf.Variable(scale, trainable=False)

    def call(self, inputs):
        """
        Normalize the inputs.

        Args:
            inputs: A tensor.

        Returns:
            The normalized input tensor.
        """
        inputs = tf.convert_to_tensor(inputs, dtype=self.dtype)
        outputs = math_ops.normalize(inputs, self.loc, self.scale)
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
        inputs = tf.convert_to_tensor(inputs, dtype=self.dtype)
        inputs = tf.cast(inputs, tf.int64)
        outputs = tf.one_hot(inputs, self.depth)
        outputs = tf.debugging.check_numerics(outputs, "outputs")
        return outputs


class AngleEncoder(tf.keras.layers.Layer):
    """
    Angle encoding layer. Encodes an angle as the cosine and sine of radians.

    Args:
        degrees (default: False):
            Whether the inputs are in degrees (True) or radians (False).
    """

    def __init__(self, degrees=False, **kwargs):
        super(AngleEncoder, self).__init__(**kwargs)
        self.degrees = degrees

    def call(self, inputs):
        inputs = tf.convert_to_tensor(inputs, dtype=self.dtype)
        if self.degrees:
            inputs = angle_ops.to_radians(inputs)
        x, y = angle_ops.to_cartesian(inputs)
        outputs = tf.concat([x, y], axis=-1)
        return outputs


class DictFeaturizer(tf.keras.layers.Layer):
    """
    Features dictionary pre-processor.

    Pre-processes a dictionary of features by passing each feature
    through a layer and concatenating the outputs.

    Args:
        feature_layers: An ordered dictionary of keys mapping to
                        Keras-compatible layers.
    """

    def __init__(self, feature_layers, **kwargs):
        super(DictFeaturizer, self).__init__(**kwargs)
        self.feature_layers = OrderedDict(feature_layers)

    def call(self, features):
        """
        Featurize a dictionary of features.

        Args:
            features: A dictionary of feautres.

        Returns:
            The concatenated outputs of each feature layer.
        """
        outputs_list = []
        for feature_key, feature_layer in self.feature_layers.items():
            inputs = features[feature_key]
            outputs = feature_layer(inputs)
            outputs_list.append(outputs)
        max_ndims = max([outputs.shape.ndims for outputs in outputs_list])
        outputs_list = [
            outputs if outputs.shape.ndims == max_ndims else outputs[..., None]
            for outputs in outputs_list
        ]
        outputs = tf.concat(outputs_list, axis=-1)
        outputs = tf.debugging.check_numerics(outputs, "outputs")
        return outputs


class ListFeaturizer(tf.keras.layers.Layer):
    """
    Features list pre-processor.

    Pre-processes a list of features by passing each feature
    through a layer and concatenating the outputs.

    Args:
        feature_layers: A list of Keras-compatible layers.
    """

    def __init__(self, feature_layers, **kwargs):
        super(ListFeaturizer, self).__init__(**kwargs)
        self.feature_layers = feature_layers

    def call(self, features):
        """
        Featurize a list of features.

        Args:
            features: A list of features.

        Returns:
            The concatenated outputs of each feature layer.
        """
        assert len(self.feature_layers) == len(
            features
        ), "must provide equal length features and feature layers"

        outputs_list = []
        for feature_layer, feature in zip(self.feature_layers, features):
            outputs = feature_layer(feature)
            outputs_list.append(outputs)
        max_ndims = max([outputs.shape.ndims for outputs in outputs_list])
        outputs_list = [
            outputs if outputs.shape.ndims == max_ndims else outputs[..., None]
            for outputs in outputs_list
        ]
        outputs = tf.concat(outputs_list, axis=-1)
        outputs = tf.debugging.check_numerics(outputs, "outputs")
        return outputs


class VecFeaturizer(tf.keras.layers.Layer):
    """
    Features vector pre-processor.

    Pre-processes a vector of features by passing each feature
    dimension through a layer and concatenating the outputs.

    Args:
        feature_layers: A list of Keras-compatible layers.
    """

    def __init__(self, feature_layers, **kwargs):
        super(VecFeaturizer, self).__init__(**kwargs)
        self.feature_layers = feature_layers

    def call(self, features):
        """
        Featurize a vector of features.

        Args:
            features: A vector of features.

        Returns:
            The concatenated outputs of each feature layer.
        """
        assert (
            len(self.feature_layers) == features.shape[-1]
        ), "must provide equal length features and feature layers"

        outputs_list = []
        for i, feature_layer in enumerate(self.feature_layers):
            outputs = feature_layer(features[..., i : i + 1])
            outputs_list.append(outputs)
        max_ndims = max([outputs.shape.ndims for outputs in outputs_list])
        outputs_list = [
            outputs if outputs.shape.ndims == max_ndims else outputs[..., None]
            for outputs in outputs_list
        ]
        outputs = tf.concat(outputs_list, axis=-1)
        outputs = tf.debugging.check_numerics(outputs, "outputs")
        return outputs


class Flatten(tf.keras.layers.Layer):
    def __init__(self, axis, **kwargs):
        super(Flatten, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        print(self.input_shape)
        outputs = tf.reshape(inputs, [])
        return outputs

    def reverse(self, inputs):
        outputs = tf.reshape(inputs, [])
        return outputs
