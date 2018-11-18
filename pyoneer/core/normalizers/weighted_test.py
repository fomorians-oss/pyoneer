import tensorflow as tf

from pyoneer.core.normalizers import weighted


class WeightedTest(tf.test.TestCase):

    def test_weighted_normalize(self):
        x = tf.constant([[[1., 1.], [1., 0.]]])
        weights = tf.constant([[[1., 1.], [1., 0.]]])
        self.assertAllClose(
            weighted.weighted_normalize(x, weights, axes=[0, 1]), 
            tf.zeros_like(x))


if __name__ == "__main__":
    tf.enable_eager_execution()
    tf.test.main()