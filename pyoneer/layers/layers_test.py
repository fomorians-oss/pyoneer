from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf

from pyoneer.layers.layers_impl import Swish, OneHotEncoder, AngleEncoder


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


if __name__ == "__main__":
    tf.test.main()
