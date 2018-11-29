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


if __name__ == '__main__':
    test.main()
