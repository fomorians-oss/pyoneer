from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_probability as tfp


class NaivePolicyStrategy(object):

    def __init__(self, policy):
        self.policy = policy

    def explore(self, i, state, action, reward, done, is_initial_state):
        policy = self.policy(state, training=False, reset_state=is_initial_state)
        return tf.cast(policy.sample(), tf.int32).numpy()

    def exploit(self, i, state, action, reward, done, is_initial_state):
        policy = self.policy(state, training=False, reset_state=is_initial_state)
        return tf.cast(policy.mode(), tf.int32).numpy()


class NaiveValueStrategy(object):

    def __init__(self, value):
        self.value = value

    def explore(self, i, state, action, reward, done, is_initial_state):
        value = self.value(state, training=False, reset_state=is_initial_state)
        return tf.cast(tfp.distributions.Categorical(logits=value).sample(), tf.int32).numpy()

    def exploit(self, i, state, action, reward, done, is_initial_state):
        value = self.value(state, training=False, reset_state=is_initial_state)
        return tf.cast(tf.argmax(value, axis=-1), tf.int32).numpy()