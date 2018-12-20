from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.losses import losses_impl
from tensorflow.python.eager import backprop

from pyoneer.rl.agents import agent_impl
from pyoneer.manip import array_ops as parray_ops
from pyoneer.math import normalization_ops

from trfl import policy_gradient_ops
from trfl import indexing_ops
from trfl import sequence_ops


def _discounted_returns(rewards, decay, weights):
    """Compute the discounted returns given the decay factor."""
    sequence_lengths = math_ops.reduce_sum(weights, axis=1)
    bootstrap_values = indexing_ops.batched_index(
        rewards, math_ops.cast(sequence_lengths - 1, dtypes.int32))
    multi_step_returns = sequence_ops.scan_discounted_sum(
        parray_ops.swap_time_major(rewards * weights), 
        parray_ops.swap_time_major(decay * weights), 
        bootstrap_values, 
        reverse=True, 
        back_prop=False)
    return parray_ops.swap_time_major(multi_step_returns)


class VanillaPolicyGradientLoss(collections.namedtuple(
    'VanillaPolicyGradientLoss', [
        'policy_gradient_loss', 
        'policy_gradient_entropy_loss', 
        'total_loss'])):
    pass


class VanillaPolicyGradientAgent(agent_impl.Agent):
    """Vanilla Policy Gradient algorithm implementation.

    Computes the policy gradient estimation.

    See this presentation by David Silver:
        http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/pg.pdf

    Example:
        ```
        class Policy(tf.keras.Model):
            def __init__(self, action_size):
                super(Policy, self).__init__()
                self.linear = tf.layers.Dense(action_size)
            def call(self, inputs):
                return tfp.distributions.MultivariateNormalDiag(self.linear(inputs))

        num_actions = 2
        policy = Policy(num_actions)
        strategy = pyrl.strategies.SampleStrategy(policy)
        agent = pyrl.agents.VanillaPolicyGradientAgent(
            policy=policy, 
            value=pynr.layers.LinearBaseline(1),
            optimizer=tf.train.GradientDescentOptimizer(1e-3))
        states, actions, rewards, weights = collect_rollouts(strategy)
        _ = agent.fit(
            states, 
            actions, 
            rewards, 
            weights, 
            decay=.999, 
            entropy_scale=.2)
        ```
    """

    def __init__(self, policy, value, optimizer):
        super(VanillaPolicyGradientAgent, self).__init__(optimizer)
        assert hasattr(value, 'fit'), '`value` must implement `fit`.'

        self.policy = policy
        self.value = value

        self.policy_gradient_loss = array_ops.constant(0.)
        self.policy_gradient_entropy_loss = array_ops.constant(0.)
        self.total_loss = array_ops.constant(0.)

    @property
    def trainable_variables(self):
        return self.policy.trainable_variables

    @property
    def loss(self):
        """Access recent losses computed after `compute_loss(...)` is called.

        Returns:
            a tuple containing `(policy_gradient_loss, policy_gradient_entropy_loss, 
                total_loss)`
        """
        return VanillaPolicyGradientLoss(
            policy_gradient_loss=self.policy_gradient_loss,
            policy_gradient_entropy_loss=self.policy_gradient_entropy_loss,
            total_loss=self.total_loss)

    def compute_loss(self, 
                     states, 
                     actions, 
                     rewards, 
                     weights, 
                     decay=.999, 
                     entropy_scale=.2, 
                     normalize_advantages=True,
                     **kwargs):
        """Computes the Vanilla PG loss.

        Args:
            states: Tensor of `[B, T, ...]` containing states.
            actions: Tensor of `[B, T, ...]` containing actions.
            rewards: Tensor of `[B, T]` containing rewards.
            weights: Tensor of shape `[B, T]` containing weights (1. or 0.).
            decay: scalar or Tensor of shape `[B, T]` containing decays/discounts.
            entropy_scale: scalar or Tensor of shape `[B, T]` containing the entropy loss scale.
            **kwargs: positional arguments (unused)

        Returns:
            the total loss Tensor of shape [].
        """
        del kwargs
        returns = _discounted_returns(rewards, decay, weights)
        self.value.fit(states, returns)

        advantages = (returns - array_ops.squeeze(self.value(states, training=True), axis=-1))
        advantages *= weights
        if normalize_advantages:
            advantages = normalization_ops.normalize_by_moments(advantages, weights)

        policy = self.policy(states, training=True)
        log_prob = policy.log_prob(actions)
        policy_gradient_loss = gen_array_ops.stop_gradient(advantages) * -log_prob
        self.policy_gradient_loss = losses_impl.compute_weighted_loss(
            policy_gradient_loss,
            weights=weights)

        entropy_loss = policy_gradient_ops.policy_entropy_loss(
            policy, 
            self.policy.trainable_variables,
            lambda policies: entropy_scale).loss
        self.policy_gradient_entropy_loss = losses_impl.compute_weighted_loss(
            entropy_loss,
            weights=weights)

        self.total_loss = math_ops.add_n([
            self.policy_gradient_loss, 
            self.policy_gradient_entropy_loss])

        return self.total_loss
