from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from pyoneer.nn import activation_ops


class NonLinearStateValue(tf.keras.Model):

    def __init__(self, state_normalizer, units=1):
        super(NonLinearStateValue, self).__init__()
        self.state_normalizer = state_normalizer
        kernel_initializer = tf.initializers.variance_scaling(scale=2.0)
        self.hidden = tf.layers.Dense(
            64, 
            activation=activation_ops.swish,
            kernel_initializer=kernel_initializer)
        self.value = tf.layers.Dense(
            units, 
            activation=None,
            kernel_initializer=kernel_initializer)

    def call(self, states, training=False, reset_state=True):
        states_norm = self.state_normalizer(states)
        hidden = self.hidden(states_norm)
        return self.value(hidden)


class NonLinearActionValue(tf.keras.Model):

    def __init__(self, state_normalizer, action_normalizer):
        super(NonLinearActionValue, self).__init__()
        self.state_normalizer = state_normalizer
        self.action_normalizer = action_normalizer
        kernel_initializer = tf.initializers.variance_scaling(scale=2.0)
        self.hidden0 = tf.layers.Dense(
            64, 
            activation=activation_ops.swish,
            kernel_initializer=kernel_initializer)
        self.hidden1 = tf.layers.Dense(
            64, 
            activation=activation_ops.swish,
            kernel_initializer=kernel_initializer)
        self.value = tf.layers.Dense(
            1, 
            activation=None,
            kernel_initializer=kernel_initializer)

    def call(self, states, actions, training=False, reset_state=True):
        states_norm = self.state_normalizer(states)
        actions_norm = self.action_normalizer(actions)
        hidden0 = self.hidden0(states_norm)
        hidden1 = self.hidden1(tf.concat([hidden0, actions_norm], axis=-1))
        return self.value(hidden1)