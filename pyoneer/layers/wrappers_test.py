from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.eager import context

from pyoneer.layers.wrappers import BatchNormBlock


class WrappersTest(tf.test.TestCase):
    def test_post_activation_batch_normalization(self):
        with context.eager_mode():
            inputs = tf.random.normal(shape=(30, 5))

            dense = tf.keras.layers.Dense(units=10, activation=None)
            bn = BatchNormBlock(dense, tf.nn.sigmoid)
            output = tf.nn.sigmoid(bn(dense(inputs), training=True))

            dense = tf.keras.layers.Dense(units=10, activation=None)
            bn = tf.keras.layers.BatchNormalization()
            expected = tf.nn.sigmoid(bn(dense(inputs), training=True))

            self.assertAllEqual(output.numpy(), expected.numpy())


if __name__ == '__main__':
    tf.test.main()
