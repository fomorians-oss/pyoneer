from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from pyoneer.manip import indexing_ops


class IndexingOpsTest(tf.test.TestCase):
    def test_batched_index(self):
        values = tf.constant([[0, 1, 2], [0, 1, 2], [0, 1, 2]])
        indices = tf.constant([0, 1, 2])
        output = indexing_ops.batched_index(values, indices)
        expected = tf.constant([0, 1, 2])
        self.assertAllEqual(output, expected)


if __name__ == "__main__":
    tf.test.main()
