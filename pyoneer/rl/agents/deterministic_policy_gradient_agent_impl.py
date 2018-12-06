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
from pyoneer.math import math_ops as pmath_ops

from trfl import policy_gradient_ops
from trfl import value_ops


class DeterministicPolicyGradientLoss(collections.namedtuple(
    'DeterministicPolicyGradientLoss', [
        'policy_gradient_loss', 
        'value_loss', 
        'total_loss'])):
    pass


class DeterministicPolicyGradientAgent(agent_impl.Agent):
    """Deterministic Policy Gradient (DPG) algorithm implementation.

    Computes the deterministic policy gradient estimation.

    Reference:
        T. P. Lillicrap, et al. "Continuous control with deep reinforcement learning".
            https://arxiv.org/abs/1509.02971
    """

    def __init__(self, policy, target_policy, value, target_value, policy_optimizer, value_optimizer):
        """Creates a new DeterministicPolicyGradientAgent.

        Args:
            policy: the target policy to optimize.
            target_policy: the target policy to optimize.
            value: the target value to optimize.
            target_value: the value to reference for TD(lambda).
            policy_optimizer: policy optimizer. Instance of `tf.train.Optimizer`.
            value_optimizer: value optimizer. Instance of `tf.train.Optimizer`.
        """
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
        """Access recent losses computed after `compute_loss(...)` is called.

        Returns:
            a tuple containing `(policy_gradient_loss, value_loss, total_loss)`
        """
        return DeterministicPolicyGradientLoss(
            policy_gradient_loss=self.policy_gradient_loss,
            value_loss=self.value_loss,
            total_loss=self.total_loss)

    def compute_loss(self, 
                     states, 
                     next_states, 
                     actions, 
                     rewards, 
                     weights, 
                     decay=.999, 
                     lambda_=1., 
                     baseline_scale=1.,
                     **kwargs):
        """Implements deep DPG loss.

        Args:
            states: Tensor of `[B, T, ...]` containing states.
            next_states: Tensor of `[B, T, ...]` containing states[t+1].
            actions: Tensor of `[B, T, ...]` containing actions.
            rewards: Tensor of `[B, T]` containing rewards.
            weights: Tensor of shape `[B, T]` containing weights (1. or 0.).
            decay: scalar or Tensor of shape `[B, T]` containing decays/discounts.
            lambda_: scalar or Tensor of shape `[B, T]` containing TD(lambda) parameter.
            baseline_scale: scalar or Tensor of shape `[B, T]` containing the baseline loss scale.
            **kwargs: positional arguments (unused)

        Returns:
            the total loss Tensor of shape [].
        """
        del kwargs
        sequence_length = math_ops.reduce_sum(weights, axis=1)
        mask = array_ops.sequence_mask(
            gen_math_ops.maximum(math_ops.cast(sequence_length, dtypes.int32) - 1, 0), 
            maxlen=states.shape[1], 
            dtype=dtypes.float32)

        policy = self.policy(states, training=True)
        target_policy = self.target_policy(next_states)

        bootstrap_value = gen_array_ops.reshape(
            self.target_value(next_states[:, -1:], target_policy[:, -1:]), 
            [-1])

        action_values = array_ops.squeeze(
            self.value(states, policy, training=True), 
            axis=-1) * mask
        self.policy_gradient_loss = losses_impl.compute_weighted_loss(
            -action_values, weights=weights)

        lambda_ = lambda_ * weights
        pcontinues = decay * weights

        baseline_loss = value_ops.td_lambda(
            parray_ops.swap_time_major(action_values), 
            parray_ops.swap_time_major(rewards),
            parray_ops.swap_time_major(pcontinues),
            gen_array_ops.stop_gradient(bootstrap_value),
            parray_ops.swap_time_major(lambda_)).loss

        self.value_loss = math_ops.reduce_mean(
            baseline_loss * baseline_scale * pmath_ops.safe_divide(1., sequence_length), 
            axis=0)

        self.total_loss = math_ops.add_n([
            self.value_loss,
            self.policy_gradient_loss])

        return self.total_loss

    def estimate_gradients(self, *args, **kwargs):
        with backprop.GradientTape(persistent=True) as tape:
            _ = self.compute_loss(*args, **kwargs)
        policy_gradients = tape.gradient(self.total_loss, self.policy.trainable_variables)
        value_gradients = tape.gradient(self.total_loss, self.value.trainable_variables)
        return (
            list(zip(policy_gradients, self.policy.trainable_variables)),
            list(zip(value_gradients, self.value.trainable_variables)))

    def fit(self, *args, **kwargs):
        policy_optimizer, value_optimizer = self.optimizer
        policy_grads_and_vars, value_grads_and_vars = self.estimate_gradients(*args, **kwargs)
        return control_flow_ops.group(
            policy_optimizer.apply_gradients(policy_grads_and_vars), 
            value_optimizer.apply_gradients(value_grads_and_vars))