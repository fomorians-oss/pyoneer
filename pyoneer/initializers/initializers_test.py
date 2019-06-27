from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from pyoneer.initializers.initializers_impl import SoftplusInverse


class InitializersTest(tf.test.TestCase):
    def test_softplus_inverse(self):
        initializer = SoftplusInverse(1.0)
        output = initializer(shape=[1, 3])
        expected = tf.constant([[0.54132485, 0.54132485, 0.54132485]], dtype=tf.float32)
        self.assertAllEqual(output, expected)


if __name__ == "__main__":
    tf.test.main()
