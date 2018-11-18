import tensorflow as tf

from pyoneer.core.normalizers import moving


class MovingTest(tf.test.TestCase):

    def test_moving_normalizer_normalize_center_scale(self):
        normalizer = moving.MovingNormalizer([1], center=True, scale=True)

        x = tf.constant([[-2.], [-1.], [0.], [1.], [2.]])
        normal_x = normalizer(x, training=True)

        mean_x, var_x = tf.nn.moments(x, axes=[0])
        std_x = tf.sqrt(var_x)
        expected_x = (x - mean_x) / std_x
        self.assertAllClose(normal_x, expected_x)

    def test_moving_normalizer_denormalize_center_scale(self):
        normalizer = moving.MovingNormalizer([1], center=True, scale=True)

        expected_x = tf.constant([[-2.], [-1.], [0.], [1.], [2.]])
        normal_x = normalizer(expected_x, training=True)
        x = normalizer.inverse(normal_x)
        self.assertAllClose(x, expected_x)


if __name__ == "__main__":
    tf.enable_eager_execution()
    tf.test.main()