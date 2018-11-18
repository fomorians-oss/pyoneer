import tensorflow as tf

from pyoneer.core.normalizers import high_low


class HighLowTest(tf.test.TestCase):

    def test_high_low_normalizer_normalize(self):
        normalizer = high_low.HighLowNormalizer(low=[-2.], high=[2.])
        
        x = [-2., -1., 0., 1., 2.]
        expected_x = [-1., -.5, 0., .5, 1.]
        self.assertAllClose(normalizer(x), expected_x)
    
    def test_high_low_normalizer_denormalize(self):
        normalizer = high_low.HighLowNormalizer(low=[-2.], high=[2.])
        
        x = [-1., -.5, 0., .5, 1.]
        expected_x = [-2., -1., 0., 1., 2.]
        self.assertAllClose(normalizer.inverse(x), expected_x)


if __name__ == "__main__":
    tf.enable_eager_execution()
    tf.test.main()