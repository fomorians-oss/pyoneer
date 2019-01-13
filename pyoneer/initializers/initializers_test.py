from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.eager import context
from tensorflow.python.platform import test

from pyoneer.initializers.init_ops import SoftplusInverse


class InitializersTest(test.TestCase):
    def test_softplus_inverse(self):
        with context.eager_mode():
            initializer = SoftplusInverse(1.0)
            output = initializer(shape=[3, 1])
            expected = tf.constant([[0.25, 0.25, 0.25]], dtype=tf.float32)
            self.assertAllEqual(output, expected)


if __name__ == '__main__':
    test.main()
