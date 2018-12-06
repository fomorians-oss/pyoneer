from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops.losses import losses_impl
from tensorflow.python.eager import backprop

from pyoneer.rl.agents import agent_impl
from pyoneer.manip import array_ops as parray_ops
from pyoneer.math import normalization_ops
from pyoneer.math import math_ops as pmath_ops

from trfl import policy_gradient_ops
from trfl import value_ops


class AdvantageActorCriticLoss(collections.namedtuple(
    'AdvantageActorCriticLoss', [
        'policy_gradient_loss', 
        'policy_gradient_entropy_loss', 
        'value_loss', 
        'total_loss'])):
    pass


class AdvantageActorCriticAgent(agent_impl.Agent):
    """Advantage Actor-Critic (A2C) algorithm implementation.

    Computes the actor-critic gradient estimation.

    See this presentation by David Silver:
        http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/pg.pdf
    """

    def __init__(self, policy, value, optimizer):
        """Creates a new AdvantageActorCriticAgent.

        Args:
            policy: the target policy to optimize.
            value: the target value to optimize.
            optimizer: Instance of `tf.train.Optimizer`.
        """
        super(AdvantageActorCriticAgent, self).__init__(optimizer)

        self.policy = policy
        self.value = value

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
            a tuple containing `(policy_gradient_loss, policy_gradient_entropy_loss, 
                value_loss, total_loss)`
        """
        return AdvantageActorCriticLoss(
            policy_gradient_loss=self.policy_gradient_loss,
            policy_gradient_entropy_loss=self.policy_gradient_entropy_loss,
            value_loss=self.value_loss,
            total_loss=self.total_loss)

    def compute_loss(self, 
                     states, 
                     actions, 
                     rewards, 
                     weights, 
                     decay=.999, 
                     lambda_=1., 
                     entropy_scale=.2, 
                     baseline_scale=1.,
                     **kwargs):
        """Computes the A2C loss.

        Args:
            states: Tensor of `[B, T, ...]` containing states.
            actions: Tensor of `[B, T, ...]` containing actions.
            rewards: Tensor of `[B, T]` containing rewards.
            weights: Tensor of shape `[B, T]` containing weights (1. or 0.).
            decay: scalar or Tensor of shape `[B, T]` containing decays/discounts.
            lambda_: scalar or Tensor of shape `[B, T]` containing generalized lambda parameter.
            entropy_scale: scalar or Tensor of shape `[B, T]` containing the entropy loss scale.
            baseline_scale: scalar or Tensor of shape `[B, T]` containing the baseline loss scale.
            **kwargs: positional arguments (unused)

        Returns:
            the total loss Tensor of shape [].
        """
        del kwargs
        sequence_lengths = math_ops.reduce_sum(weights, axis=1)

        policy = self.policy(states, training=True)
        baseline_values = array_ops.squeeze(self.value(states, training=True), axis=-1)

        pcontinues = decay * weights
        bootstrap_values = baseline_values[:, -1]
        baseline_loss, td_lambda = value_ops.td_lambda(
            parray_ops.swap_time_major(baseline_values), 
            parray_ops.swap_time_major(rewards), 
            parray_ops.swap_time_major(pcontinues), 
            bootstrap_values, 
            lambda_)

        advantages = parray_ops.swap_time_major(td_lambda.temporal_differences)
        advantages = normalization_ops.weighted_moments_normalize(advantages, weights)
        advantages = gen_array_ops.stop_gradient(advantages)

        log_prob = policy.log_prob(actions)
        log_prob = parray_ops.expand_to(log_prob, ndims=3)
        policy_gradient_loss = advantages * -math_ops.reduce_sum(log_prob, axis=-1)
        self.policy_gradient_loss = losses_impl.compute_weighted_loss(
            policy_gradient_loss,
            weights=weights)

        entropy_loss = policy_gradient_ops.policy_entropy_loss(
            policy, 
            self.policy.trainable_variables,
            lambda policies: entropy_scale).loss
        entropy_loss = parray_ops.expand_to(entropy_loss, ndims=3)
        entropy_loss = math_ops.reduce_sum(entropy_loss, axis=-1)
        self.policy_gradient_entropy_loss = losses_impl.compute_weighted_loss(
            entropy_loss,
            weights=weights)

        self.value_loss = math_ops.reduce_mean(
            baseline_loss * baseline_scale * pmath_ops.safe_divide(1., sequence_lengths),
            axis=0)

        self.total_loss = math_ops.add_n([
            self.value_loss,
            self.policy_gradient_loss, 
            self.policy_gradient_entropy_loss])

        return self.total_loss
