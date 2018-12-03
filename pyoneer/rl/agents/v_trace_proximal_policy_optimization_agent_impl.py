from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.eager import backprop
from tensorflow.python.ops.losses import losses_impl

from pyoneer.rl.agents import agent_impl
from pyoneer.manip import array_ops as parray_ops
from pyoneer.math import normalization_ops

from trfl import policy_gradient_ops
from trfl import vtrace_ops


class VTraceProximalPolicyOptimizationLoss(collections.namedtuple(
    'VTraceProximalPolicyOptimizationLoss', [
        'policy_gradient_loss', 
        'policy_gradient_entropy_loss', 
        'value_loss', 
        'total_loss'])):
    pass


class VTraceProximalPolicyOptimizationAgent(agent_impl.Agent):
    """PPO with V-trace (IMPALA) algorithm implementation.

    Computes the PPO with V-trace return targets (IMPALA) gradient estimation:
    """

    def __init__(self, policy, behavioral_policy, value, optimizer):
        super(VTraceProximalPolicyOptimizationAgent, self).__init__(optimizer)

        self.policy = policy
        self.behavioral_policy = behavioral_policy
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
        return VTraceProximalPolicyOptimizationLoss(
            policy_gradient_loss=self.policy_gradient_loss,
            policy_gradient_entropy_loss=self.policy_gradient_entropy_loss,
            value_loss=self.value_loss,
            total_loss=self.total_loss)

    def compute_loss(self, rollouts, decay=.999, lambda_=1., entropy_scale=.2, baseline_scale=1., ratio_epsilon=.2):
        policy = self.policy(rollouts.states, training=True)
        behavioral_policy = self.behavioral_policy(rollouts.states)
        baseline_values = parray_ops.swap_time_major(
            array_ops.squeeze(self.value(rollouts.states, training=True), axis=-1))

        pcontinues = parray_ops.swap_time_major(decay * rollouts.weights)
        bootstrap_values = baseline_values[-1, :]

        log_prob = policy.log_prob(rollouts.actions)
        behavioral_log_prob = behavioral_policy.log_prob(rollouts.actions)
        log_rhos = parray_ops.swap_time_major(log_prob - gen_array_ops.stop_gradient(behavioral_log_prob))
        vtrace_returns = vtrace_ops.vtrace_from_importance_weights(
            log_rhos,
            pcontinues,
            parray_ops.swap_time_major(rollouts.rewards),
            baseline_values,
            bootstrap_values)

        advantages = parray_ops.swap_time_major(vtrace_returns.pg_advantages)
        advantages = normalization_ops.weighted_moments_normalize(advantages, rollouts.weights)
        advantages = gen_array_ops.stop_gradient(advantages)

        ratio = parray_ops.swap_time_major(gen_math_ops.exp(log_rhos))
        clipped_ratio = clip_ops.clip_by_value(ratio, 1. - ratio_epsilon, 1. + ratio_epsilon)

        self.policy_gradient_loss = -losses_impl.compute_weighted_loss(
            gen_math_ops.minimum(advantages * ratio, advantages * clipped_ratio), 
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
            math_ops.multiply(
                math_ops.reduce_sum(math_ops.square(vtrace_returns.vs - baseline_values), axis=0), 
                baseline_scale * .5),
            axis=0)

        self.total_loss = math_ops.add_n([
            self.value_loss,
            self.policy_gradient_loss, 
            self.policy_gradient_entropy_loss])

        return self.total_loss
