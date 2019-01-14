from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.eager import context
from tensorflow.python.platform import test

from pyoneer.nn import moments_impl


class MomentsTest(test.TestCase):
    def test_streaming_moments(self):
        with context.eager_mode():
            moments = moments_impl.StreamingMoments(shape=[3])

            # sample 1
            inputs = tf.constant([[[-1.0, 0.0, +1.0]]])
            weights = tf.constant([[[1.0, 0.0, 1.0]]])

            moments(inputs, weights=weights, training=True)

            expected_mean = tf.constant([-1.0, 0.0, +1.0], dtype=tf.float32)
            expected_var = tf.constant([0.0, 0.0, 0.0], dtype=tf.float32)
            expected_std = tf.sqrt(expected_var)

            self.assertAllClose(expected_mean, moments.mean.numpy())
            self.assertAllClose(expected_var, moments.variance.numpy())
            self.assertAllClose(expected_std, moments.std.numpy())

            # sample 2
            inputs = tf.constant([[[0.0, +1.0, -1.0]]])
            weights = tf.constant([[[0.0, 1.0, 1.0]]])

            moments(inputs, weights=weights, training=True)

            expected_mean = tf.constant([-1.0, +1.0, 0.0], dtype=tf.float32)
            expected_var = tf.constant([0.0, 0.0, 2.0], dtype=tf.float32)
            expected_std = tf.sqrt(expected_var)

            self.assertAllClose(expected_mean, moments.mean.numpy())
            self.assertAllClose(expected_var, moments.variance.numpy())
            self.assertAllClose(expected_std, moments.std.numpy())

            # sample 3
            inputs = tf.constant([[[+1.0, -1.0, 0.0]]])
            weights = tf.constant([[[1.0, 1.0, 0.0]]])

            moments(inputs, weights=weights, training=True)

            expected_mean = tf.constant([0.0, 0.0, 0.0], dtype=tf.float32)
            expected_var = tf.constant([2.0, 2.0, 2.0], dtype=tf.float32)
            expected_std = tf.sqrt(expected_var)

            self.assertAllClose(expected_mean, moments.mean.numpy())
            self.assertAllClose(expected_var, moments.variance.numpy())
            self.assertAllClose(expected_std, moments.std.numpy())


if __name__ == '__main__':
    test.main()
