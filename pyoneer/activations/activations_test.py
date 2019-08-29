from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from pyoneer.activations.activations_impl import swish


class ActivationsTest(tf.test.TestCase):
    def test_swish(self):
        x = tf.constant([-1.0, 0.0, +1.0])
        actual = swish(x)
        expected = x * tf.nn.sigmoid(x)
        self.assertAllClose(actual, expected)


if __name__ == "__main__":
    tf.test.main()
