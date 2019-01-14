from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.eager import context
from tensorflow.python.platform import test

from pyoneer.manip import array_ops

# class ArrayOpsTest(test.TestCase):
#     def test_pad_or_truncate(self):
#         with context.eager_mode():
#             x = tf.constant([[0, 1, 2]])
#             actual = array_ops.pad_or_truncate(
#                 x, maxsize=4, axis=1, constant_values=3)
#             expected = tf.constant([[0, 1, 2, 3]])
#             self.assertAllClose(actual, expected)

#             x = tf.constant([[0, 1, 2, 3, 4]])
#             actual = array_ops.pad_or_truncate(
#                 x, maxsize=4, axis=1, constant_values=3)
#             self.assertAllClose(actual, expected)

#     def test_shift(self):
#         with context.eager_mode():
#             x = tf.constant([[0, 1, 2]])
#             actual = array_ops.shift(x, axis=1, shift=1, constant_values=3)
#             expected = tf.constant([[0, 1, 2, 3]])
#             self.assertAllClose(actual, expected)

#             x = tf.constant([[0, 1, 2, 3, 4]])
#             actual = array_ops.shift(x, axis=1, shift=1, constant_values=3)
#             self.assertAllClose(actual, expected)

if __name__ == '__main__':
    test.main()
