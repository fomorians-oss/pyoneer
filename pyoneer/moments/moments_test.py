from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from pyoneer.moments import moments_impl


class MomentsTest(tf.test.TestCase):
    def test_range_moments(self):
        mean, variance = moments_impl.range_moments(
            minval=tf.constant([-2.0]), maxval=tf.constant([2.0])
        )
        expected_mean = tf.constant([0.0], dtype=tf.float32)
        expected_variance = tf.constant([4.0], dtype=tf.float32)
        self.assertAllEqual(mean, expected_mean)
        self.assertAllEqual(variance, expected_variance)

    def test_static_moments(self):
        mean = tf.constant([-1.0, 0.0, +1.0], dtype=tf.float32)
        variance = tf.constant([1.0, 2.0, 1.0], dtype=tf.float32)
        moments = moments_impl.StaticMoments(mean, variance)

        expected_mean = tf.constant([-1.0, 0.0, +1.0], dtype=tf.float32)
        expected_var = tf.constant([1.0, 2.0, 1.0], dtype=tf.float32)
        expected_std = tf.sqrt(expected_var)

        self.assertAllClose(expected_mean, moments.mean.numpy())
        self.assertAllClose(expected_var, moments.variance.numpy())
        self.assertAllClose(expected_std, moments.std.numpy())

    def test_streaming_moments(self):
        moments = moments_impl.StreamingMoments(shape=[3])

        # sample 1
        inputs = tf.constant([[[-1.0, 0.0, +1.0]]])
        sample_weight = tf.constant([[[1.0, 0.0, 1.0]]])

        moments.update_state(inputs, sample_weight=sample_weight)

        expected_mean = tf.constant([-1.0, 0.0, +1.0], dtype=tf.float32)
        expected_var = tf.constant([0.0, 0.0, 0.0], dtype=tf.float32)
        expected_std = tf.sqrt(expected_var)

        self.assertAllClose(expected_mean, moments.mean.numpy())
        self.assertAllClose(expected_var, moments.variance.numpy())
        self.assertAllClose(expected_std, moments.std.numpy())

        # sample 2
        inputs = tf.constant([[[0.0, +1.0, -1.0]]])
        sample_weight = tf.constant([[[0.0, 1.0, 1.0]]])

        moments.update_state(inputs, sample_weight=sample_weight)

        expected_mean = tf.constant([-1.0, +1.0, 0.0], dtype=tf.float32)
        expected_var = tf.constant([0.0, 0.0, 2.0], dtype=tf.float32)
        expected_std = tf.sqrt(expected_var)

        self.assertAllClose(expected_mean, moments.mean.numpy())
        self.assertAllClose(expected_var, moments.variance.numpy())
        self.assertAllClose(expected_std, moments.std.numpy())

        # sample 3
        inputs = tf.constant([[[+1.0, -1.0, 0.0]]])
        sample_weight = tf.constant([[[1.0, 1.0, 0.0]]])

        moments.update_state(inputs, sample_weight=sample_weight)

        expected_mean = tf.constant([0.0, 0.0, 0.0], dtype=tf.float32)
        expected_var = tf.constant([2.0, 2.0, 2.0], dtype=tf.float32)
        expected_std = tf.sqrt(expected_var)

        self.assertAllClose(expected_mean, moments.mean.numpy())
        self.assertAllClose(expected_var, moments.variance.numpy())
        self.assertAllClose(expected_std, moments.std.numpy())

    def test_exponential_moving_moments(self):
        moments = moments_impl.ExponentialMovingMoments(shape=[3], rate=0.9)

        # sample 1
        inputs = tf.constant([[[-1.0, 0.0, +1.0]]])
        sample_weight = tf.constant([[[1.0, 0.0, 1.0]]])

        moments.update_state(inputs, sample_weight=sample_weight)

        expected_mean = tf.constant([-1.0, 0.0, +1.0], dtype=tf.float32)
        expected_var = tf.constant([0.0, 0.0, 0.0], dtype=tf.float32)
        expected_std = tf.sqrt(expected_var)

        self.assertAllClose(expected_mean, moments.mean.numpy())
        self.assertAllClose(expected_var, moments.variance.numpy())
        self.assertAllClose(expected_std, moments.std.numpy())

        # sample 2
        inputs = tf.constant([[[0.0, +1.0, -1.0]]])
        sample_weight = tf.constant([[[0.0, 1.0, 1.0]]])

        moments.update_state(inputs, sample_weight=sample_weight)

        expected_mean = tf.constant([-1.0, 1.0, 0.8], dtype=tf.float32)
        expected_var = tf.constant([0.0, 0.0, 0.0], dtype=tf.float32)
        expected_std = tf.sqrt(expected_var)

        self.assertAllClose(expected_mean, moments.mean.numpy())
        self.assertAllClose(expected_var, moments.variance.numpy())
        self.assertAllClose(expected_std, moments.std.numpy())

        # sample 3
        inputs = tf.constant([[[+1.0, -1.0, 0.0]]])
        sample_weight = tf.constant([[[1.0, 1.0, 0.0]]])

        moments.update_state(inputs, sample_weight=sample_weight)

        expected_mean = tf.constant([-0.8, 0.8, 0.8], dtype=tf.float32)
        expected_var = tf.constant([0.0, 0.0, 0.0], dtype=tf.float32)
        expected_std = tf.sqrt(expected_var)

        self.assertAllClose(expected_mean, moments.mean.numpy())
        self.assertAllClose(expected_var, moments.variance.numpy())
        self.assertAllClose(expected_std, moments.std.numpy())


if __name__ == "__main__":
    tf.test.main()
