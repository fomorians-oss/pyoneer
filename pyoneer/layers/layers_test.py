from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf

from pyoneer.math import angle_ops, math_ops
from pyoneer.layers.layers_impl import (
    Swish,
    OneHotEncoder,
    AngleEncoder,
    Nest,
    ConcreteDropout,
)


class LayersTest(tf.test.TestCase):
    def test_swish_layer(self):
        layer = Swish()
        inputs = tf.constant([-1.0, 0.0, +1.0], dtype=tf.float32)
        outputs = layer(inputs)
        expected = tf.constant([-0.268941, 0.0, 0.731059], dtype=tf.float32)
        self.assertAllClose(outputs, expected)

    def test_one_hot_encoder(self):
        layer = OneHotEncoder(depth=4)
        inputs = tf.constant([3], dtype=tf.int32)
        outputs = layer(inputs)
        expected = tf.constant([[0.0, 0.0, 0.0, 1.0]], dtype=tf.float32)
        self.assertAllEqual(outputs, expected)

    def test_angle_encoder_radians(self):
        layer = AngleEncoder(degrees=False)
        inputs = tf.constant([[math.pi]], dtype=tf.float32)
        outputs = layer(inputs)
        expected = tf.constant([[-1, 0]], dtype=tf.float32)
        self.assertAllClose(outputs, expected)

    def test_angle_encoder_degrees(self):
        layer = AngleEncoder(degrees=True)
        inputs = tf.constant([[180]], dtype=tf.float32)
        outputs = layer(inputs)
        expected = tf.constant([[-1, 0]], dtype=tf.float32)
        self.assertAllClose(outputs, expected)

    def test_nest_features(self):
        nest_layer = Nest(
            {
                "category": OneHotEncoder(depth=3),
                "angle": AngleEncoder(degrees=True),
                "nested": {
                    "category": OneHotEncoder(depth=3),
                    "angle": AngleEncoder(degrees=True),
                },
            }
        )
        features = {
            "category": tf.constant([1]),
            "angle": tf.constant([[-45.0]]),
            "nested": {"category": tf.constant([2]), "angle": tf.constant([[45.0]])},
        }
        outputs = nest_layer(features)
        expected = tf.stack(
            [
                [
                    tf.math.cos(angle_ops.to_radians(-45.0)),
                    tf.math.sin(angle_ops.to_radians(-45.0)),
                    0.0,
                    1.0,
                    0.0,
                    tf.math.cos(angle_ops.to_radians(+45.0)),
                    tf.math.sin(angle_ops.to_radians(+45.0)),
                    0.0,
                    0.0,
                    1.0,
                ]
            ],
            axis=0,
        )
        self.assertAllClose(outputs, expected)

    def test_concrete_dropout(self):
        dropout_layer = ConcreteDropout(regularizer_scale=1.0)
        self.assertAllClose(dropout_layer.rate, 0.1)

        inputs = tf.constant([[-1.0, 0.0, +1.0]], dtype=tf.float32)
        outputs = dropout_layer(inputs)
        expected = tf.constant([[-1.111111, 0.0, 1.111111]], dtype=tf.float32)
        self.assertAllClose(outputs, expected)

        self.assertAllClose(dropout_layer.losses[0], -0.325083)
        dropout_layer.rate_logit.assign(math_ops.sigmoid_inverse(0.2))
        self.assertAllClose(dropout_layer.losses[0], -0.500403)


if __name__ == "__main__":
    tf.test.main()
