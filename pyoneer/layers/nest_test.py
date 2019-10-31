from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from pyoneer.math import angle_ops
from pyoneer.layers import layers_impl
from pyoneer.layers import nest_impl


class NestTest(tf.test.TestCase):

    def testMapStructure(self):
        map_layer = nest_impl.MapStructure(layers={
                "category": layers_impl.OneHotEncoder(depth=3),
                "angle": layers_impl.AngleEncoder(degrees=True),
                "nested": {
                    "category": layers_impl.OneHotEncoder(depth=3),
                    "angle": layers_impl.AngleEncoder(degrees=True),
                },
            }
        )
        features = {
            "category": tf.constant([1]),
            "angle": tf.constant([[-45.0]]),
            "nested": {
                "category": tf.constant([2]),
                "angle": tf.constant([[45.0]])
            },
        }
        outputs = map_layer(features)

        expected = {
            "category": tf.constant([[0, 1, 0]]),
            "angle": tf.stack([
                tf.math.cos(angle_ops.to_radians([-45.0])),
                tf.math.sin(angle_ops.to_radians([-45.0]))],
                axis=-1),
            "nested": {
                "category": tf.constant([[0, 0, 1]], tf.dtypes.int64),
                "angle": tf.stack([
                    tf.math.cos(angle_ops.to_radians([45.0])),
                    tf.math.sin(angle_ops.to_radians([45.0]))],
                    axis=-1),
            },
        }
        tf.nest.map_structure(self.assertAllEqual, outputs, expected)

    def testReduceStructure(self):
        map_layer = nest_impl.MapStructure(
            layers={
                "category": layers_impl.OneHotEncoder(depth=3),
                "angle": layers_impl.AngleEncoder(degrees=True),
                "nested": {
                    "category": layers_impl.OneHotEncoder(depth=3),
                    "angle": layers_impl.AngleEncoder(degrees=True),
                },
            }
        )
        reduce_layer = nest_impl.ReduceStructure(
            reduction=tf.keras.layers.Concatenate(axis=-1))

        features = {
            "category": tf.constant([1]),
            "angle": tf.constant([[-45.0]]),
            "nested": {
                "category": tf.constant([2]),
                "angle": tf.constant([[45.0]])
            },
        }
        outputs = map_layer(features)
        outputs = reduce_layer(outputs)

        mapped = {
            "category": tf.constant([[0., 1., 0.]]),
            "angle": tf.stack([
                tf.math.cos(angle_ops.to_radians([-45.0])),
                tf.math.sin(angle_ops.to_radians([-45.0]))],
                axis=-1),
            "nested": {
                "category": tf.constant([[0., 0., 1.]]),
                "angle": tf.stack([
                    tf.math.cos(angle_ops.to_radians([45.0])),
                    tf.math.sin(angle_ops.to_radians([45.0]))],
                    axis=-1),
            },
        }
        expected = tf.concat(tf.nest.flatten(mapped), axis=-1)
        tf.nest.map_structure(self.assertAllEqual, outputs, expected)



if __name__ == "__main__":
    tf.test.main()