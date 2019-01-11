from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.eager as tfe

from tensorflow.python.eager import context
from tensorflow.python.platform import test

from pyoneer import losses


class L2RegularizationTest(test.TestCase):
    def test_l2_regularization(self):
        with context.eager_mode():
            trainable_variables = [
                tfe.Variable(tf.random.uniform((2, 3)), trainable=True),
                tfe.Variable(tf.random.uniform((3, 4)), trainable=True)
            ]

            scale = 0.5
            output = losses.l2_regularization(trainable_variables, scale)
            expected = scale * tf.reduce_sum(
                [tf.reduce_sum(v**2) / 2.0 for v in trainable_variables])

            self.assertAllEqual(output, expected)


if __name__ == '__main__':
    test.main()
