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

from trfl import policy_gradient_ops
from trfl import value_ops


class AdvantageActorCriticLoss(collections.namedtuple(
    'AdvantageActorCriticLoss', [
        'policy_gradient_loss', 'policy_gradient_entropy_loss', 'value_loss', 'total_loss'])):
    pass


class AdvantageActorCriticAgent(agent_impl.Agent):
    """Advantage Actor-Critic (A2C) algorithm implementation.

    Computes the actor-critic gradient estimation:

        A2C ~= 𝔼[A(s, a)▽log(π(a|s))]
        A(s, a) ~= 𝔼[R - V(s)] 
        V(s) ~= |R - V(s)|^2

    """

    def __init__(self, policy, value, optimizer):
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
        return AdvantageActorCriticLoss(
            policy_gradient_loss=self.policy_gradient_loss,
            policy_gradient_entropy_loss=self.policy_gradient_entropy_loss,
            value_loss=self.value_loss,
            total_loss=self.total_loss)

    def compute_loss(self, rollouts, decay=.999, lambda_=1., entropy_scale=.2, baseline_scale=1.):
        policy = self.policy(rollouts.states, training=True)
        baseline_values = array_ops.squeeze(self.value(rollouts.states, training=True), axis=-1)

        pcontinues = decay * rollouts.weights
        bootstrap_values = baseline_values[:, -1]
        baseline_loss, td_lambda = value_ops.td_lambda(
            parray_ops.swap_time_major(baseline_values), 
            parray_ops.swap_time_major(rollouts.rewards), 
            parray_ops.swap_time_major(pcontinues), 
            bootstrap_values, 
            lambda_)

        advantages = parray_ops.swap_time_major(td_lambda.temporal_differences)
        advantages = normalization_ops.weighted_moments_normalize(advantages, rollouts.weights)
        advantages = gen_array_ops.stop_gradient(advantages)

        log_prob = policy.log_prob(rollouts.actions)
        log_prob = parray_ops.expand_to(log_prob, ndims=3)
        policy_gradient_loss = advantages * -math_ops.reduce_sum(log_prob, axis=-1)
        self.policy_gradient_loss = losses_impl.compute_weighted_loss(
            policy_gradient_loss,
            weights=rollouts.weights)

        entropy_loss = policy_gradient_ops.policy_entropy_loss(
            policy, 
            self.policy.trainable_variables,
            lambda policies: entropy_scale).loss
        entropy_loss = parray_ops.expand_to(entropy_loss, ndims=3)
        entropy_loss = math_ops.reduce_sum(entropy_loss, axis=-1)
        self.policy_gradient_entropy_loss = losses_impl.compute_weighted_loss(
            entropy_loss,
            weights=rollouts.weights)

        self.value_loss = math_ops.reduce_mean(
            math_ops.multiply(baseline_loss, baseline_scale),
            axis=0)

        self.total_loss = math_ops.add_n([
            self.value_loss,
            self.policy_gradient_loss, 
            self.policy_gradient_entropy_loss])

        return self.total_loss
