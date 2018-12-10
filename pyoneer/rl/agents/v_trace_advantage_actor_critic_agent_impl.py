from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.eager import backprop
from tensorflow.python.ops.losses import losses_impl

from pyoneer.rl.agents import agent_impl
from pyoneer.manip import array_ops as parray_ops
from pyoneer.math import normalization_ops
from pyoneer.math import math_ops as pmath_ops

from trfl import policy_gradient_ops
from trfl import vtrace_ops
from trfl import indexing_ops


class VTraceAdvantageActorCriticLoss(collections.namedtuple(
    'VTraceAdvantageActorCriticLoss', [
        'policy_gradient_loss', 
        'policy_gradient_entropy_loss', 
        'value_loss', 
        'total_loss'])):
    pass


class VTraceAdvantageActorCriticAgent(agent_impl.Agent):
    """A2C with V-trace (IMPALA) algorithm implementation.

    Computes A2C with V-trace return targets (IMPALA) gradient estimation.

    Reference:
        L. Espeholt, et al. "IMPALA: Scalable Distributed Deep-RL with Importance Weighted 
            Actor-Learner Architectures". https://arxiv.org/abs/1802.01561
    
    Example:
        ```
        class Policy(tf.keras.Model):
            def __init__(self, action_size):
                super(Policy, self).__init__()
                self.linear = tf.layers.Dense(action_size)
            def call(self, inputs):
                return tfp.distributions.MultivariateNormalDiag(self.linear(inputs))

        class Value(tf.keras.Model):
            def __init__(self, num_units):
                super(Value, self).__init__()
                self.linear = tf.layers.Dense(num_units)
            def call(self, inputs):
                return self.linear(inputs)

        num_actions = 2
        behavioral_policy = Policy(num_actions)
        strategy = pyrl.strategies.SampleStrategy(behavioral_policy)
        agent = pyrl.agents.VTraceAdvantageActorCriticAgent(
            policy=Policy(num_actions), 
            behavioral_policy=behavioral_policy, 
            value=Value(1),
            optimizer=tf.train.GradientDescentOptimizer(1e-3))
        states, actions, rewards, weights = collect_rollouts(strategy)
        _ = agent.fit(
            states, 
            actions, 
            rewards, 
            weights, 
            decay=.999, 
            lambda_=1., 
            entropy_scale=.2, 
            baseline_scale=1.)
        trfl.update_target_variables(
            agent.behavioral_policy.trainable_variables,
            agent.policy.trainable_variables)
        ```
    """

    def __init__(self, policy, behavioral_policy, value, optimizer):
        super(VTraceAdvantageActorCriticAgent, self).__init__(optimizer)

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
        """Access recent losses computed after `compute_loss(...)` is called.

        Returns:
            a tuple containing `(policy_gradient_loss, policy_gradient_entropy_loss, 
                value_loss, total_loss)`
        """
        return VTraceAdvantageActorCriticLoss(
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
                     normalize_advantages=True,
                     **kwargs):
        """Computes the A2C with V-trace return targets (IMPALA) loss.

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
        sequence_length = math_ops.reduce_sum(weights, axis=1)
        total_num = math_ops.reduce_sum(sequence_length)

        policy = self.policy(states, training=True)
        behavioral_policy = self.behavioral_policy(states)
        baseline_values = array_ops.squeeze(
            self.value(states, training=True), 
            axis=-1) * weights
        bootstrap_values = indexing_ops.batched_index(
            baseline_values, math_ops.cast(sequence_length - 1, dtypes.int32))
        baseline_values = parray_ops.swap_time_major(baseline_values)

        pcontinues = parray_ops.swap_time_major(decay * weights)
        log_prob = policy.log_prob(actions)
        log_rhos = parray_ops.swap_time_major(log_prob) - parray_ops.swap_time_major(
            behavioral_policy.log_prob(actions))
        vtrace_returns = vtrace_ops.vtrace_from_importance_weights(
            log_rhos,
            pcontinues,
            parray_ops.swap_time_major(rewards),
            baseline_values,
            bootstrap_values)

        advantages = parray_ops.swap_time_major(vtrace_returns.pg_advantages)
        if normalize_advantages:
            advantages = normalization_ops.weighted_moments_normalize(advantages, weights)
        advantages = gen_array_ops.stop_gradient(advantages)

        policy_gradient_loss = advantages * -log_prob
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

        baseline_loss = math_ops.reduce_sum(
            math_ops.square(vtrace_returns.vs - baseline_values), axis=0)
        self.value_loss = pmath_ops.safe_divide(
            .5 * baseline_scale * math_ops.reduce_sum(baseline_loss), total_num)
        self.value_loss = gen_array_ops.check_numerics(
            self.value_loss, 'value_loss')

        self.total_loss = math_ops.add_n([
            self.value_loss,
            self.policy_gradient_loss, 
            self.policy_gradient_entropy_loss])

        return self.total_loss
