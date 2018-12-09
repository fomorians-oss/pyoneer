from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
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
from trfl import indexing_ops
from trfl import base_ops


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
        policy = Policy(num_actions)
        strategy = pyrl.strategies.SampleStrategy(policy)
        agent = pyrl.agents.AdvantageActorCriticAgent(
            policy=policy, 
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
        ```
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
                     normalize_advantages=True,
                     **kwargs):
        """Computes the A2C loss.

        Args:
            states: Tensor of `[B, T, ...]` containing states.
            actions: Tensor of `[B, T, ...]` containing actions.
            rewards: Tensor of `[B, T]` containing rewards.
            weights: Tensor of shape `[B, T]` containing weights 
                (1. or 0.).
            decay: scalar or Tensor of shape `[B, T]` containing 
                decays/discounts.
            lambda_: scalar or Tensor of shape `[B, T]` containing 
                generalized lambda parameter.
            entropy_scale: scalar or Tensor of shape `[B, T]` containing 
                the entropy loss scale.
            baseline_scale: scalar or Tensor of shape `[B, T]` containing 
                the baseline loss scale.
            **kwargs: positional arguments (unused)

        Returns:
            the total loss Tensor of shape [].

        Raises:
            ValueError: If tensors are empty or fail the rank and mutual
                compatibility asserts.
        """
        del kwargs
        base_ops.assert_rank_and_shape_compatibility([weights], 2)
        sequence_lengths = math_ops.reduce_sum(weights, axis=1)
        total_num = math_ops.reduce_sum(sequence_lengths)

        baseline_values = array_ops.squeeze(
            self.value(states, training=True), 
            axis=-1) * weights
        base_ops.assert_rank_and_shape_compatibility([rewards, baseline_values], 2)

        pcontinues = decay * weights
        lambda_ = lambda_ * weights
        bootstrap_values = indexing_ops.batched_index(
            baseline_values, math_ops.cast(sequence_lengths - 1, dtypes.int32))

        baseline_loss, td_lambda = value_ops.td_lambda(
            parray_ops.swap_time_major(baseline_values), 
            parray_ops.swap_time_major(rewards), 
            parray_ops.swap_time_major(pcontinues), 
            bootstrap_values, 
            parray_ops.swap_time_major(lambda_))

        advantages = parray_ops.swap_time_major(td_lambda.temporal_differences)
        if normalize_advantages:
            advantages = normalization_ops.weighted_moments_normalize(advantages, weights)
        advantages = gen_array_ops.check_numerics(advantages, 'advantages')

        policy = self.policy(states, training=True)
        log_prob = policy.log_prob(actions)
        policy_gradient_loss =  gen_array_ops.stop_gradient(advantages) * -log_prob
        self.policy_gradient_loss = losses_impl.compute_weighted_loss(
            policy_gradient_loss,
            weights=weights)
        self.policy_gradient_loss = gen_array_ops.check_numerics(
            self.policy_gradient_loss, 'policy_gradient_loss')

        entropy_loss = policy_gradient_ops.policy_entropy_loss(
            policy, 
            self.policy.trainable_variables,
            lambda policies: entropy_scale).loss
        self.policy_gradient_entropy_loss = losses_impl.compute_weighted_loss(
            entropy_loss,
            weights=weights)
        self.policy_gradient_entropy_loss = gen_array_ops.check_numerics(
            self.policy_gradient_entropy_loss, 'policy_gradient_entropy_loss')

        self.value_loss = pmath_ops.safe_divide(
            baseline_scale * math_ops.reduce_sum(baseline_loss), total_num)
        self.value_loss = gen_array_ops.check_numerics(
            self.value_loss, 'value_loss')

        self.total_loss = math_ops.add_n([
            self.value_loss,
            self.policy_gradient_loss, 
            self.policy_gradient_entropy_loss])

        return self.total_loss


class MultiAdvantageActorCriticAgent(AdvantageActorCriticAgent):
    """Advantage Actor-Critic (A2C) with multiple-values implementation.

    Computes the actor-critic gradient estimation.

    Example:
        ```
        import tensorflow as tf
        import tensorflow_probability as tfp
        import pyoneer.rl as pyrl

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
        policy = Policy(num_actions)
        strategy = pyrl.strategies.SampleStrategy(policy)
        agent = pyrl.agents.MultiAdvantageActorCriticAgent(
            policy=policy, 
            value=Value(2),
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
        ```
    """

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
        """Computes the A2C loss with multiple values.

        Args:
            states: Tensor of `[B, T, ...]` containing states.
            actions: Tensor of `[B, T, ...]` containing actions.
            rewards: Tensor of `[B, T, V]` containing rewards.
            weights: Tensor of shape `[B, T]` containing weights (1. or 0.).
            decay: scalar, 1-D Tensor of shape [V], or Tensor of shape 
                `[B, T]` or `[B, T, V]` containing decays/discounts.
            lambda_: scalar, 1-D Tensor of shape [V], or Tensor of shape 
                `[B, T]` or `[B, T, V]` containing generalized lambda parameter.
            entropy_scale: scalar or Tensor of shape `[B, T]` containing the entropy loss scale.
            baseline_scale: scalar or Tensor of shape `[B, T]` containing the baseline loss scale.
            **kwargs: positional arguments (unused)

        Returns:
            the total loss Tensor of shape [].
        """
        del kwargs
        base_ops.assert_rank_and_shape_compatibility([weights], 2)
        sequence_lengths = math_ops.reduce_sum(weights, axis=1)
        total_num = math_ops.reduce_sum(sequence_lengths)

        multi_advantages = []
        self.value_loss = []
        multi_baseline_values = self.value(states, training=True) * array_ops.expand_dims(weights, axis=-1)

        base_ops.assert_rank_and_shape_compatibility(
            [rewards, multi_baseline_values], 3)
        multi_baseline_values = array_ops.unstack(multi_baseline_values, axis=-1)
        num_values = len(multi_baseline_values)

        base_shape = rewards.shape
        decay = self._least_fit(decay, base_shape)
        lambda_ = self._least_fit(lambda_, base_shape)
        baseline_scale = self._least_fit(baseline_scale, base_shape)

        for i in range(num_values):
            pcontinues = decay[..., i] * weights
            lambdas = lambda_[..., i] * weights
            bootstrap_values = indexing_ops.batched_index(
                multi_baseline_values[i], math_ops.cast(sequence_lengths - 1, dtypes.int32))
            baseline_loss, td_lambda = value_ops.td_lambda(
                parray_ops.swap_time_major(multi_baseline_values[i]), 
                parray_ops.swap_time_major(rewards[..., i]), 
                parray_ops.swap_time_major(pcontinues), 
                bootstrap_values, 
                parray_ops.swap_time_major(lambdas))
            value_loss = pmath_ops.safe_divide(
                baseline_scale[i] * math_ops.reduce_sum(baseline_loss), total_num)
            self.value_loss.append(
                gen_array_ops.check_numerics(value_loss, 'value_loss'))
            advantages = parray_ops.swap_time_major(td_lambda.temporal_differences)
            multi_advantages.append(advantages)

        advantages = math_ops.add_n(multi_advantages) # A = A[0] + A[1] + ...
        if normalize_advantages:
            advantages = normalization_ops.weighted_moments_normalize(advantages, weights)
        advantages = gen_array_ops.stop_gradient(advantages)

        policy = self.policy(states, training=True)
        log_prob = policy.log_prob(actions)
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

        self.total_loss = math_ops.add_n([
            math_ops.add_n(self.value_loss),
            self.policy_gradient_loss, 
            self.policy_gradient_entropy_loss])

        return self.total_loss
