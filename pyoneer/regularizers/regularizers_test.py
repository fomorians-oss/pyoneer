from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from pyoneer.layers.layers_impl import ConcreteDropout
from pyoneer.regularizers.regularizers_impl import DropoutL2


class RegularizersTest(tf.test.TestCase):
    def test_dropout_l2(self):
        dropout_layer = ConcreteDropout()
        regularizer = DropoutL2(dropout_layer, scale=1.0)
        kernel = tf.ones(shape=[3, 1])
        outputs = regularizer(kernel)
        expected = 3.333333
        self.assertAllClose(outputs, expected)

        dropout_layer.rate_logit.assign_add(0.1)
        outputs = regularizer(kernel)
        expected = 3.36839
        self.assertAllClose(outputs, expected)


if __name__ == "__main__":
    tf.test.main()
