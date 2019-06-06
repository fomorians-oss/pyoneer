from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from pyoneer.math import logical_ops


class LogicalOpsTest(tf.test.TestCase):
    def test_isclose(self):
        x = tf.constant([0.9, 1.0, 1.1, 1.2])
        actual_x = logical_ops.isclose(x, 1.0, rtol=1e-5, atol=0.1)
        expected_x = tf.constant([True, True, True, False])
        self.assertAllEqual(actual_x, expected_x)


if __name__ == "__main__":
    tf.test.main()
