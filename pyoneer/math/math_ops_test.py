from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.eager import context
from tensorflow.python.platform import test

from pyoneer.math import math_ops


class MathOpsTest(test.TestCase):
    def test_safe_divide(self):
        with context.eager_mode():
            x = tf.ones(shape=[7], dtype=tf.float32)
            y = tf.constant([-1.0, -0.5, -0.2, 0.0, +0.2, +0.5, +1.0])
            actual = math_ops.safe_divide(x, y)
            expected = tf.constant([-1.0, -2.0, -5.0, 1.0, +5.0, +2.0, +1.0])
            self.assertAllEqual(actual, expected)

    def test_rescale(self):
        with context.eager_mode():
            x = tf.constant([0.5, 0.5, 0.5])
            actual = math_ops.rescale(x, 0.0, 1.0, -1.0, 1.0)
            expected = tf.zeros(shape=[3], dtype=tf.float32)
            self.assertAllEqual(actual, expected)

    def test_normalize(self):
        with context.eager_mode():
            x = tf.constant([-1.0, 0.0, 1.0, 0.0])
            weights = tf.constant([1.0, 1.0, 1.0, 0.0])
            actual = math_ops.normalize(x, loc=-1.0, scale=2.0, weights=weights)
            expected = tf.constant([0.0, 0.5, 1.0, 0.0])
            self.assertAllClose(actual, expected)

    def test_denormalize(self):
        with context.eager_mode():
            x = tf.constant([0.0, 0.5, 1.0, 0.0])
            weights = tf.constant([1.0, 1.0, 1.0, 0.0])
            actual = math_ops.denormalize(x, loc=-1.0, scale=2.0, weights=weights)
            expected = tf.constant([-1.0, 0.0, 1.0, 0.0])
            self.assertAllClose(actual, expected)


if __name__ == "__main__":
    test.main()
