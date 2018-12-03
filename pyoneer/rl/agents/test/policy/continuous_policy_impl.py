from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.eager as tfe
import tensorflow_probability as tfp

from pyoneer.layers import rnn_impl
from pyoneer.nn import activation_ops
from pyoneer.initializers import init_ops


class ContinuousPolicy(tf.keras.Model):

    def __init__(self, state_normalizer, output_size, scale=1., activation=None):
        super(ContinuousPolicy, self).__init__()
        self.state_normalizer = state_normalizer
        kernel_initializer = tf.initializers.variance_scaling(scale=2.0)
        self.hidden = tf.layers.Dense(
            64, 
            activation=activation_ops.swish,
            kernel_initializer=kernel_initializer)
        self.loc = tf.layers.Dense(
            output_size, 
            activation=activation,
            kernel_initializer=kernel_initializer)
        self.scale = tfe.Variable(
            initial_value=init_ops.InverseSoftplusScale(scale=scale)(
                shape=[output_size]))

    def call(self, inputs, training=False, reset_state=False):
        inputs_norm = self.state_normalizer(inputs)
        hidden = self.hidden(inputs_norm)
        loc = self.loc(hidden)
        scale = tf.nn.softplus(self.scale)
        return tfp.distributions.MultivariateNormalDiag(loc=loc, scale_diag=scale)


class RecurrentContinuousPolicy(tf.keras.Model):

    def __init__(self, state_normalizer, output_size, scale=1., activation=None):
        super(RecurrentContinuousPolicy, self).__init__()
        self.state_normalizer = state_normalizer
        kernel_initializer = tf.initializers.variance_scaling(scale=2.0)
        self.hidden = tf.layers.Dense(
            64, 
            activation=activation_ops.swish,
            kernel_initializer=kernel_initializer)
        self.rnn = rnn_impl.RNN(tf.nn.rnn_cell.LSTMCell(64))
        self.loc = tf.layers.Dense(
            output_size, 
            activation=activation,
            kernel_initializer=kernel_initializer)
        self.scale = tfe.Variable(
            initial_value=init_ops.InverseSoftplusScale(scale=scale)(
                shape=[output_size]))

    def call(self, inputs, training=False, reset_state=True):
        inputs_norm = self.state_normalizer(inputs)
        hidden = self.hidden(inputs_norm)
        hidden = self.rnn(hidden, reset_state=reset_state)
        loc = self.loc(hidden)
        scale = tf.nn.softplus(self.scale)
        return tfp.distributions.MultivariateNormalDiag(loc=loc, scale_diag=scale)

