from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from pyoneer.layers import self_attention_impl


class SelfAttentionTest(tf.test.TestCase):

    def testOutputShapes(self):
        tf.random.set_seed(32)
        self_attention = self_attention_impl.SelfAttentionQKV(8, 4, 2)
        inputs = tf.random.uniform([10, 5, 12], -1., 1.)
        outputs = self_attention(inputs)
        self.assertAllEqual(tf.shape(outputs), [10, 5, 4 * 2])


if __name__ == "__main__":
    tf.test.main()