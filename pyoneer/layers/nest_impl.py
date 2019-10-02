from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class MapStructure(tf.keras.Model):

    def __init__(self, layers=None, layer=None, **kwargs):
        """
        Maps an arbitrarily nested structure of layers to an equivalent
        structure of input features using the `tf.nest` API.

        Example:

            ```python
            # Apply a function to each value.
            map_layer = MapStructure(layers={
                "category": OneHotEncoder(depth=3),
                "angle": AngleEncoder(degrees=True),
                "nested": {
                    "category": OneHotEncoder(depth=3),
                    "angle": AngleEncoder(degrees=True),
                }
            })

            # Or apply one function to all values.
            map_layer = MapStructure(layer=AngleEncoder(degrees=True))

            features = {
                "category": tf.constant([1]),
                "angle": tf.constant([[-45.0]]),
                "nested": {
                    "category": tf.constant([2]),
                    "angle": tf.constant([[+45.0]]),
                }
            }
            outputs = map_layer(features)
            ```

        Args:
            layers: An arbitrarily nested structure of layers.
        """
        super(MapStructure, self).__init__(**kwargs)
        layer_is_none = (layer is None)
        layers_is_none = (layers is None)

        if (not layer_is_none) and (not layers_is_none):
            raise ValueError('Only one of `layer` or `layers` can be supplied.')
        if layer_is_none and layers_is_none:
            raise ValueError('One of `layer` or `layers` has to be supplied.')

        self.nested_layer = layer
        self.nested_layers = layers

    def call(self, inputs):
        def call_layer(inputs, layer=self.nested_layer):
            return layer(inputs)

        inputs_mapped = tf.nest.map_structure(
            call_layer,
            *((inputs,) if self.nested_layers is None else (inputs, self.nested_layers)))
        return inputs_mapped


class ReduceStructure(tf.keras.layers.Layer):

    def __init__(self, reduction, **kwargs):
        """
        Maps an arbitrarily nested structure of layers to an equivalent
        structure of input features using the `tf.nest` API.

        Example:

            ```python
            map_layer = MapStructure(
                layers={
                    "category": OneHotEncoder(depth=3),
                    "angle": AngleEncoder(degrees=True),
                    "nested": {
                        "category": OneHotEncoder(depth=3),
                        "angle": AngleEncoder(degrees=True),
                    }
                })
            reduce_layer = ReduceStructure(
                reduction=tf.keras.layers.Concatenate(axis=-1))
            features = {
                "category": tf.constant([1]),
                "angle": tf.constant([[-45.0]]),
                "nested": {
                    "category": tf.constant([2]),
                    "angle": tf.constant([[+45.0]]),
                }
            }
            outputs = reduce_layer(map_layer(features))
            ```

        Args:
            reduction: A function that takes a list of tensors
                and returns a single tensor.
        """
        super(ReduceStructure, self).__init__(**kwargs)
        self.reduction = reduction

    def call(self, inputs_mapped):
        inputs_flattened = tf.nest.flatten(inputs_mapped)
        outputs = self.reduction(inputs_flattened)
        return outputs
