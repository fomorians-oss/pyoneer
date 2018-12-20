from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import context
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_impl
from tensorflow.python.platform import test

from pyoneer.features import normalizer_impl


class NormalizerTest(test.TestCase):

    def test_sample_average_normalizer_normalize_center_scale(self):
        with context.eager_mode():
            normalizer = normalizer_impl.SampleAverageNormalizer([1], center=True, scale=True)

            x = array_ops.constant([[-2.], [-1.], [0.], [1.], [2.]])
            normal_x = normalizer(x, training=True)

            mean_x, var_x = nn_impl.moments(x, axes=[0])
            std_x = math_ops.sqrt(var_x)
            expected_x = (x - mean_x) / std_x
            self.assertAllClose(normal_x, expected_x)

    def test_sample_average_normalizer_denormalize_center_scale(self):
        with context.eager_mode():
            normalizer = normalizer_impl.SampleAverageNormalizer([1], center=True, scale=True)

            expected_x = array_ops.constant([[-2.], [-1.], [0.], [1.], [2.]])
            normal_x = normalizer(expected_x, training=True)
            x = normalizer.inverse(normal_x)
            self.assertAllClose(x, expected_x)

    def test_high_low_normalizer_normalize(self):
        with context.eager_mode():
            normalizer = normalizer_impl.HighLowNormalizer(minval=[-2.], maxval=[2.])
            
            x = [-2., -1., 0., 1., 2.]
            expected_x = [-1., -.5, 0., .5, 1.]
            self.assertAllClose(normalizer(x), expected_x)

    def test_high_low_normalizer_denormalize(self):
        with context.eager_mode():
            normalizer = normalizer_impl.HighLowNormalizer(minval=[-2.], maxval=[2.])
            
            x = [-1., -.5, 0., .5, 1.]
            expected_x = [-2., -1., 0., 1., 2.]
            self.assertAllClose(normalizer.inverse(x), expected_x)


if __name__ == "__main__":
    test.main()