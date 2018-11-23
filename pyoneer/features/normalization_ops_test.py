from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.eager import context
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test

from pyoneer.features import normalization_ops


class NormalizationOpsTest(test.TestCase):

    def test_weighted_moments_normalize(self):
        with context.eager_mode():
            x = [[[1., 1.], [1., 0.]]]
            weights = [[[1., 1.], [1., 0.]]]
            self.assertAllClose(
                normalization_ops.weighted_moments_normalize(x, weights, axes=[0, 1]), 
                array_ops.zeros_like(x))


if __name__ == "__main__":
    test.main()