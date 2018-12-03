from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.eager import backprop
from tensorflow.python.ops.losses import losses_impl
from tensorflow.python.training import optimizer

from pyoneer.rl.agents import agent_impl
from pyoneer.manip import array_ops as parray_ops

from trfl import policy_gradient_ops
from trfl import value_ops


class DeterministicPolicyGradientLoss(collections.namedtuple(
    'DeterministicPolicyGradientLoss', [
        'policy_gradient_loss', 'value_loss', 'total_loss'])):
    pass


class DeterministicPolicyGradientAgent(agent_impl.Agent):
    """Deterministic Policy Gradient (DPG) algorithm implementation.

    Computes the deterministic policy gradient estimation:
    """

    def __init__(self, policy, target_policy, value, target_value, policy_optimizer, value_optimizer):
        super(DeterministicPolicyGradientAgent, self).__init__((policy_optimizer, value_optimizer))

        self.policy = policy
        self.target_policy = target_policy
        self.value = value
        self.target_value = target_value

        self.policy_gradient_loss = array_ops.constant(0.)
        self.policy_gradient_entropy_loss = array_ops.constant(0.)
        self.value_loss = array_ops.constant(0.)
        self.total_loss = array_ops.constant(0.)

    @property
    def trainable_variables(self):
        return self.policy.trainable_variables + self.value.trainable_variables

    @property
    def loss(self):
        return DeterministicPolicyGradientLoss(
            policy_gradient_loss=self.policy_gradient_loss,
            value_loss=self.value_loss,
            total_loss=self.total_loss)

    def compute_loss(self, rollouts, delay=.999, lambda_=1., entropy_scale=.2):
        sequence_length = math_ops.cast(
            math_ops.reduce_sum(rollouts.weights, axis=-1), 
            dtypes.int32)

        mask = array_ops.sequence_mask(
            gen_math_ops.maximum(sequence_length - 1, 0), 
            maxlen=rollouts.states.shape[1], 
            dtype=dtypes.float32)

        policy = self.policy(rollouts.states, training=True)
        target_policy = self.target_policy(rollouts.next_states)

        bootstrap_state = rollouts.next_states[:, -1:]
        bootstrap_action = target_policy.mode()[:, -1:]
        bootstrap_value = array_ops.squeeze(self.target_value(bootstrap_state, bootstrap_action))

        action_values = array_ops.squeeze(self.value(rollouts.states, policy, training=True), axis=-1) * mask
        self.policy_gradient_loss = losses_impl.compute_weighted_loss(
            -action_values, weights=rollouts.weights)

        lambda_ = lambda_ * rollouts.weights
        pcontinues = delay * rollouts.weights

        self.value_loss = math_ops.reduce_mean(
            value_ops.td_lambda(
                parray_ops.swap_time_major(action_values), 
                parray_ops.swap_time_major(rollouts.rewards),
                parray_ops.swap_time_major(pcontinues),
                gen_array_ops.stop_gradient(bootstrap_value),
                parray_ops.swap_time_major(lambda_)).loss,
            axis=0)

        self.total_loss = math_ops.add_n([
            self.value_loss,
            self.policy_gradient_loss])

        return self.total_loss

    def estimate_gradients(self, rollouts, **kwargs):
        with backprop.GradientTape(persistent=True) as tape:
            _ = self.compute_loss(rollouts, **kwargs)
        policy_gradients = tape.gradient(self.total_loss, self.policy.trainable_variables)
        value_gradients = tape.gradient(self.total_loss, self.value.trainable_variables)
        return (
            list(zip(policy_gradients, self.policy.trainable_variables)),
            list(zip(value_gradients, self.value.trainable_variables)))

    def fit(self, rollouts, **kwargs):
        policy_optimizer, value_optimizer = self.optimizer
        policy_grads_and_vars, value_grads_and_vars = self.estimate_gradients(rollouts, **kwargs)
        return control_flow_ops.group(
            policy_optimizer.apply_gradients(policy_grads_and_vars), 
            value_optimizer.apply_gradients(value_grads_and_vars))