from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from pyoneer.nn import activation_ops


class ActivationOpsTest(tf.test.TestCase):
    def test_swish(self):
        x = tf.constant([-1.0, 0.0, +1.0])
        actual = activation_ops.swish(x)
        expected = tf.constant([-0.268941, 0.0, 0.731059])
        self.assertAllClose(actual, expected)


if __name__ == "__main__":
    tf.test.main()
