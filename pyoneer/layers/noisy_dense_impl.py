import math

from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import nn
from tensorflow.python.layers import base


class NoisyDense(base.Layer):
    """Noisy, dense layer class. 
    
    This layer applies factorized noise to the weights and biases. 
    
    Reference:
        Fortunato et al., "Noisy Networks for Exploration". 
            https://arxiv.org/abs/1706.10295

    Arguments:
        num_units: The number of outputs.
        sigma0: The standard deviation for the weight noise.
        activation: Activation function (callable). Set it to None to maintain a
            linear activation.
        use_bias: Boolean, whether the layer uses a bias.
        kernel_initializer: Initializer function for the weight matrix.
            If `None` (default), weights are initialized using the default
            initializer used by `tf.get_variable`.
        bias_initializer: Initializer function for the bias.
        kernel_regularizer: Regularizer function for the weight matrix.
        bias_regularizer: Regularizer function for the bias.
        activity_regularizer: Regularizer function for the output.
        kernel_constraint: An optional projection function to be applied to the
            kernel after being updated by an `Optimizer` (e.g. used to implement
            norm constraints or value constraints for layer weights). The function
            must take as input the unprojected variable and must return the
            projected variable (which must have the same shape). Constraints are
            not safe to use when doing asynchronous distributed training.
        bias_constraint: An optional projection function to be applied to the
            bias after being updated by an `Optimizer`.
        kernel_trainable: Boolean, kernel should be added to the graph collection
            `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
        trainable: Boolean, if `True` also add variables to the graph collection
            `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
        name: String, the name of the layer. Layers with the same name will
            share weights, but to avoid mistakes we require reuse=True in such cases.
        reuse: Boolean, whether to reuse the weights of a previous layer
            by the same name.
    """

    def __init__(self,
                num_units,
                sigma0=0.5,
                activation=None,
                use_bias=True,
                kernel_initializer=None,
                bias_initializer=init_ops.zeros_initializer(),
                kernel_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                kernel_constraint=None,
                bias_constraint=None,
                kernel_trainable=True,
                trainable=True,
                name=None,
                **kwargs):
        super(NoisyDense, self).__init__(trainable=trainable, name=name,
                                        activity_regularizer=activity_regularizer,
                                        **kwargs)
        self.num_units = num_units
        self.sigma0 = ops.convert_to_tensor(sigma0, dtype=dtypes.float32)
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint
        self.kernel_trainable = kernel_trainable
        self.input_spec = base.InputSpec(min_ndim=2)

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        if input_shape[-1].value is None:
            raise ValueError('The last dimension of the inputs to `Dense` '
                            'should be defined. Found `None`.')
        input_dim = input_shape[-1].value
        stddev = 1. / math_ops.sqrt(math_ops.cast(input_dim, dtypes.float32)) * self.sigma0

        self.input_spec = base.InputSpec(min_ndim=2,
                                         axes={-1: input_dim})

        self.kernel_loc = self.add_variable('kernel_loc',
                                            shape=[input_dim, self.num_units],
                                            initializer=self.kernel_initializer,
                                            regularizer=self.kernel_regularizer,
                                            constraint=self.kernel_constraint,
                                            dtype=self.dtype,
                                            trainable=self.kernel_trainable)
        self.kernel_scale = self.add_variable('kernel_scale',
                                              shape=[input_dim, self.num_units],
                                              initializer=init_ops.truncated_normal_initializer(
                                                  stddev=stddev),
                                              regularizer=self.kernel_regularizer,
                                              constraint=self.kernel_constraint,
                                              dtype=self.dtype,
                                              trainable=self.kernel_trainable)
        self.kernel_epsilon = _factorized_noise(input_dim, self.num_units)

        if self.use_bias:
            self.bias_loc = self.add_variable('bias_loc',
                                            shape=[self.num_units],
                                            initializer=self.bias_initializer,
                                            regularizer=self.bias_regularizer,
                                            constraint=self.bias_constraint,
                                            dtype=self.dtype,
                                            trainable=True)
            self.bias_scale = self.add_variable('bias_scale',
                                                shape=[self.num_units],
                                                initializer=init_ops.truncated_normal_initializer(
                                                    stddev=stddev),
                                                regularizer=self.bias_regularizer,
                                                constraint=self.bias_constraint,
                                                dtype=self.dtype,
                                                trainable=True)
            self.bias_epsilon = random_ops.random_normal([self.num_units], dtype=self.bias_scale.dtype)

        else:
            self.bias = None
            self.built = True

    def call(self, inputs):
        inputs = ops.convert_to_tensor(inputs, dtype=self.dtype)
        shape = inputs.get_shape().as_list()
        kernel = gen_math_ops.add(
            self.kernel_loc, self.kernel_scale * self.kernel_epsilon)

        if len(shape) > 2:
            # Broadcasting is required for the inputs.
            outputs = standard_ops.tensordot(inputs, kernel, [[len(shape) - 1],
                                                             [0]])
            # Reshape the output back to the original ndim of the input.
            if not context.executing_eagerly():
                outputs.set_shape(shape)
        else:
            outputs = math_ops.matmul(inputs, kernel)
        if self.use_bias:
            bias = gen_math_ops.add(
                self.bias_loc, self.bias_scale * self.bias_epsilon)
            outputs = nn.bias_add(outputs, bias)

        if self.activation is not None:
            return self.activation(outputs)  # pylint: disable=not-callable
        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        if input_shape[-1] is None:
            raise ValueError(
                'The innermost dimension of input_shape must be defined, but saw: %s'
                % input_shape)
        return input_shape[:-1].concatenate(self.num_units)


def _signed_sqrt(values):
    return math_ops.sqrt(math_ops.abs(values)) * math_ops.sign(values)


def _factorized_noise(inputs, outputs):
    noise1 = _signed_sqrt(random_ops.random_normal((inputs, 1)))
    noise2 = _signed_sqrt(random_ops.random_normal((1, outputs)))
    return math_ops.matmul(noise1, noise2)
