from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf

from pyoneer.math import angle_ops


class AnglesOpsTest(tf.test.TestCase):
    def test_to_radians(self):
        inputs = tf.constant([-360, -180, -90, 0, +90, +180, +360], dtype=tf.float32)
        outputs = angle_ops.to_radians(inputs)
        expected = tf.constant(
            [
                -math.pi * 2,
                -math.pi,
                -math.pi / 2,
                0,
                +math.pi / 2,
                +math.pi,
                +math.pi * 2,
            ],
            dtype=tf.float32,
        )
        self.assertAllEqual(outputs, expected)

    def test_to_degrees(self):
        inputs = tf.constant(
            [
                -math.pi * 2,
                -math.pi,
                -math.pi / 2,
                0,
                +math.pi / 2,
                +math.pi,
                +math.pi * 2,
            ],
            dtype=tf.float32,
        )
        outputs = angle_ops.to_degrees(inputs)
        expected = tf.constant([-360, -180, -90, 0, +90, +180, +360], dtype=tf.float32)
        self.assertAllEqual(outputs, expected)

    def test_to_cartesian(self):
        inputs = tf.constant(
            [
                -math.pi * 2,
                -math.pi,
                -math.pi / 2,
                0,
                +math.pi / 2,
                +math.pi,
                +math.pi * 2,
            ],
            dtype=tf.float32,
        )
        outputs_x, outputs_y = angle_ops.to_cartesian(inputs)
        expected_x = tf.constant([+1, -1, +0, +1, +0, -1, +1], dtype=tf.float32)
        expected_y = tf.constant([+0, +0, -1, +0, +1, +0, +0], dtype=tf.float32)
        self.assertAllClose(outputs_x, expected_x)
        self.assertAllClose(outputs_y, expected_y)

    def test_to_polar(self):
        inputs_x = tf.constant([+1, -1, +0, +1, +0, -1, +1], dtype=tf.float32)
        inputs_y = tf.constant([+0, +0, -1, +0, +1, +0, +0], dtype=tf.float32)
        outputs_rho, outputs_phi = angle_ops.to_polar(inputs_x, inputs_y)
        expected_rho = tf.constant([1, 1, 1, 1, 1, 1, 1], dtype=tf.float32)
        expected_phi = tf.constant(
            [0, math.pi, -math.pi / 2, 0, math.pi / 2, math.pi, 0], dtype=tf.float32
        )
        self.assertAllClose(outputs_rho, expected_rho)
        self.assertAllClose(outputs_phi, expected_phi)


if __name__ == "__main__":
    tf.test.main()
