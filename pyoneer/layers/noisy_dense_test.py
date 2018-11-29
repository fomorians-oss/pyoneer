import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import test

from pyoneer.layers import noisy_dense_impl


class NoisyDenseTest(test.TestCase):

    def test_NoisyDense(self):
        with context.eager_mode():
            inputs = array_ops.zeros([1, 2])
            layer = noisy_dense_impl.NoisyDense(2, use_bias=False)
            outputs = layer(inputs)
            self.assertAllClose(outputs, array_ops.constant([[0., 0.]]))


if __name__ == '__main__':
    test.main()