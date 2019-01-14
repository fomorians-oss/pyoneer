from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.eager import context
from tensorflow.python.platform import test

from pyoneer import indexing_ops


class IndexingOpsTest(test.TestCase):
    def test_batched_index(self):
        with context.eager_mode():
            values = tf.constant([[0, 1, 2], [0, 1, 2], [0, 1, 2]])
            indices = tf.constant([0, 1, 2])
            output = indexing_ops.batched_index(values, indices)
            expected = tf.constant([0, 1, 2])
            self.assertAllEqual(output, expected)


if __name__ == '__main__':
    test.main()
