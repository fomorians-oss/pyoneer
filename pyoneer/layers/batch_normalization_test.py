from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from copy import deepcopy
from tensorflow.python.eager import context

from pyoneer.layers.batch_normalization import BatchNormBlock


class BatchNormBlockTest(tf.test.TestCase):
    def test_batchnorm_block(self):
        with context.eager_mode():
            inputs = tf.random.normal(shape=(30, 15), mean=5.0, stddev=10.0)
            # Initialize called layers separately
            dense = tf.keras.layers.Dense(units=20, activation=None)
            bn = tf.keras.layers.BatchNormalization()
            sigmoid = tf.keras.layers.Activation('sigmoid')
            seq_output = sigmoid(bn(dense(inputs), training=True))
            # Initialize BatchNormBlock layer, copying the individual layers
            # from above to set the BatchNormBlock's layers so the initialized
            # weights are the same.
            bn_block = BatchNormBlock(deepcopy(dense), 'sigmoid')
            bn_block.bn = deepcopy(bn)
            block_output = bn_block(inputs, training=True)
            # Assert the two outputs are equal
            self.assertAllEqual(seq_output.numpy(), block_output.numpy())


if __name__ == '__main__':
    tf.test.main()
