from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_probability as tfp

from pyoneer.nn import activation_ops
from pyoneer.layers import rnn_impl


class DiscretePolicy(tf.keras.Model):

    def __init__(self, state_normalizer, output_size):
        super(DiscretePolicy, self).__init__()
        self.state_normalizer = state_normalizer
        kernel_initializer = tf.initializers.variance_scaling(scale=2.0)
        self.hidden = tf.layers.Dense(
            64, 
            activation=activation_ops.swish,
            kernel_initializer=kernel_initializer)
        self.logits = tf.layers.Dense(
            output_size, 
            activation=None,
            kernel_initializer=kernel_initializer)

    def call(self, inputs, training=False, reset_state=False):
        inputs_norm = self.state_normalizer(inputs)
        hidden = self.hidden(inputs_norm)
        logits = self.logits(hidden)
        return tfp.distributions.Categorical(logits=logits)


class RecurrentDiscretePolicy(tf.keras.Model):

    def __init__(self, state_normalizer, output_size):
        super(RecurrentDiscretePolicy, self).__init__()
        self.state_normalizer = state_normalizer
        kernel_initializer = tf.initializers.variance_scaling(scale=2.0)
        self.hidden = tf.layers.Dense(
            64, 
            activation=activation_ops.swish,
            kernel_initializer=kernel_initializer)
        self.rnn = rnn_impl.RNN(tf.nn.rnn_cell.LSTMCell(64))
        self.logits = tf.layers.Dense(
            output_size, 
            activation=None,
            kernel_initializer=kernel_initializer)

    def call(self, inputs, training=False, reset_state=True):
        inputs_norm = self.state_normalizer(inputs)
        hidden = self.hidden(inputs_norm)
        hidden = self.rnn(hidden, reset_state=reset_state)
        logits = self.logits(hidden)
        return tfp.distributions.Categorical(logits=logits)

