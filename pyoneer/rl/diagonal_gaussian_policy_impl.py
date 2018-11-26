from tensorflow.python import layers
from tensorflow.python.framweork import ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops.resource_variable_ops import ResourceVariable as Variable

from tensorflow_probability.python import distributions

from pyoneer.rl import policy_impl
from pyoneer.rl import init_ops


class DiagonalGaussianPolicy(policy_impl.Policy):

    def __init__(self, 
                 units=None, 
                 loc=None, 
                 scale=None, 
                 activation=None,
                 kernel_initializer=None,
                 bias_initializer=None,
                 scale_initializer=init_ops.SoftplusScale(scale=1.)):
        super(DiagonalGaussianPolicy, self).__init__()
        if units is not None:
            self.loc = layers.Dense(
                units, 
                name='loc', 
                activation=activation,
                bias_initializer=bias_initializer,
                kernel_initializer=kernel_initializer)
            self.scale = Variable(
                initial_value=scale_initializer(units),
                name='scale')
        elif isinstance(loc, ops.Tensor) and isinstance(scale, ops.Tensor):
            self.loc = loc
            self.scale = scale
        else:
            raise NotImplementedError()

    def call(self, inputs):
        loc = self.loc(inputs)
        scale = gen_nn_ops.softplus(self.scale)
        return distributions.MultivariateNormalDiag(loc=loc, scale_diag=scale)

    @staticmethod
    def from_parameters(loc, scale):
        return DiagonalGaussianPolicy(loc=loc, scale=scale)