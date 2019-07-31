from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from pyoneer.layers.dropout_impl import ConcreteDropout


class DropoutTest(tf.test.TestCase):
    def test_concrete_dropout(self):
        base_layer = tf.keras.layers.Dense(units=3, activation=None)
        dropout_layer = ConcreteDropout(base_layer)
        inputs = tf.constant([[-1.0, 0.0, +1.0]], dtype=tf.float32)
        outputs = dropout_layer(inputs)
        expected = tf.constant([[0.35601515, 1.2371143, 0.75698006]], dtype=tf.float32)
        self.assertAllClose(dropout_layer.rate, 0.1)
        self.assertAllClose(outputs, expected)


if __name__ == "__main__":
    tf.test.main()
