from tensorflow.python import layers
from tensorflow.python.framweork import ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops.resource_variable_ops import ResourceVariable as Variable

from tensorflow_probability.python import distributions

from pyoneer.rl import policy_impl


class RelaxedCategoricalPolicy(policy_impl.Policy):

    def __init__(self, 
                 units=None,
                 logits=None, 
                 temperature=None, 
                 activation=None,
                 kernel_initializer=None,
                 bias_initializer=None,
                 temperature_initializer=init_ops.RandomUniform(.25, .75)):
        super(RelaxedCategoricalPolicy, self).__init__()
        if isinstance(logits, ops.Tensor) and isinstance(temperature, ops.Tensor):
            self.logits = logits
            self.temperature = temperature
        else:
            assert units is not None
            self.logits = layers.Dense(
                units, 
                name='logits', 
                activation=activation,
                bias_initializer=bias_initializer,
                kernel_initializer=kernel_initializer)
            self.temperature = Variable(
                initial_value=temperature_initializer([]),
                name='temperature')

    def call(self, inputs):
        logits = self.logits(inputs)
        return distributions.RelaxedOneHotCategorical(self.temperature, logits=logits)

    @staticmethod
    def from_parameters(logits, temperature):
        return RelaxedCategoricalPolicy(logits=logits, temperature=temperature)
