from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python import layers
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import control_flow_ops

from pyoneer.manip import array_ops as parray_ops


class LinearBaseline(layers.Layer):
    """An unbiased linear approximator.

    The intention is to be used as a baseline to reduce variance without introducing bias.
        >>> A(S, a) = Q(S, a) - b(S)
        >>> b(S) = S@W, where ~= |R - b(s)|^2
    """

    def __init__(self, units, l2_regularizer=1e-5):
        """Creates a new LinearBaseline.

        Args:
            units: the number of outputs.
            l2_regularizer: the coeffecient of regularization.
        """
        super(LinearBaseline, self).__init__()
        self.l2_regularizer = l2_regularizer
        self.units = units

    def build(self, input_shape):
        self.linear = self.add_variable(
            name='linear',
            shape=[input_shape[-1], self.units],
            initializer=init_ops.Zeros(), 
            dtype=dtypes.float32,
            trainable=False)
        self.built = True

    def fit(self, inputs, outputs):
        """Solves a regularized least-squares problem given inputs and outputs.

        Args:
            inputs: the input Tensor.
            outputs: the desired output Tensor.
        
        Returns:
            Op that assigns the new linear transformation with the result.
        """
        if not self.built:
            self.build(inputs.shape)
        inputs = parray_ops.flatten(inputs)
        outputs = gen_array_ops.reshape(outputs, [-1, self.units])
        outputs = linalg_ops.matrix_solve_ls(inputs, outputs, l2_regularizer=self.l2_regularizer)
        return state_ops.assign(self.linear, outputs)

    def call(self, inputs, training=False):
        k = inputs.shape[0]
        inputs = parray_ops.flatten(inputs)
        baseline = inputs @ self.linear
        return gen_array_ops.reshape(baseline, [k, -1])