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


class LinearBaseline(layers.Layer):

    """An unbiased linear approximator.
    
    Notes:
        The intention is to be used as a baseline to reduce variance without introducing bias.
            >>> A(s, a) = Q(S, a) - b(S)
            >>> b(s) = S@W, where ~= |R - b(s)|^2
    """

    def __init__(self, l2_regularizer=1e-5):
        super(LinearBaseline, self).__init__()
        self.l2_regularizer = l2_regularizer

    def build(self, input_shape):
        self.linear = self.add_variable(
            name='linear',
            shape=[input_shape[-1], 1],
            initializer=init_ops.Zeros(), 
            dtype=dtypes.float32,
            trainable=False)

    def fit(self, inputs, outputs):
        outputs = gen_array_ops.reshape(outputs, [-1, 1])
        outputs = linalg_ops.matrix_solve_ls(inputs, outputs, l2_regularizer=self.l2_regularizer)
        with ops.control_dependencies([state_ops.assign(self.linear, outputs)]):
            return array_ops.constant(0.)

    def call(self, inputs, training=False):
        k = inputs.shape[0]
        inputs = gen_array_ops.reshape(inputs, [-1, inputs.shape[-1]])
        baseline = gen_math_ops.matmul(inputs, self.linear)
        return gen_array_ops.reshape(baseline, [k, -1])