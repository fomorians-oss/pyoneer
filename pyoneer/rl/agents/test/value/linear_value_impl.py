from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from pyoneer.layers import linear_baseline_impl


class LinearStateValue(tf.keras.Model):

    def __init__(self, state_normalizer):
        super(LinearStateValue, self).__init__()
        self.state_normalizer = state_normalizer
        self.linear = linear_baseline_impl.LinearBaseline(1, l2_regularizer=1e-5)

    def fit(self, states, returns):
        states_norm = self.state_normalizer(states)
        return self.linear.fit(states_norm, returns)

    def call(self, states, training=False, reset_state=True):
        states_norm = self.state_normalizer(states)
        return self.linear(states_norm, training=training)


class LinearActionValue(tf.keras.Model):

    def __init__(self, state_normalizer, action_normalizer):
        super(LinearActionValue, self).__init__()
        self.state_normalizer = state_normalizer
        self.action_normalizer = action_normalizer
        self.linear = linear_baseline_impl.LinearBaseline(1, l2_regularizer=1e-5)

    def fit(self, states, actions, returns):
        states_norm = self.state_normalizer(states)
        actions_norm = self.action_normalizer(actions)
        states_and_actions_norm = tf.concat([states_norm, actions_norm], axis=-1)
        return self.linear.fit(states_and_actions_norm, returns)

    def call(self, states, actions, training=False, reset_state=True):
        states_norm = self.state_normalizer(states)
        actions_norm = self.action_normalizer(actions)
        states_and_actions_norm = tf.concat([states_norm, actions_norm], axis=-1)
        return self.linear(states_and_actions_norm, training=training)
