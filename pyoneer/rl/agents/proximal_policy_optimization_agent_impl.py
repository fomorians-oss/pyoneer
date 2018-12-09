from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.eager import backprop
from tensorflow.python.ops.losses import losses_impl

from pyoneer.rl.agents import agent_impl
from pyoneer.manip import array_ops as parray_ops
from pyoneer.math import normalization_ops
from pyoneer.math import math_ops as pmath_ops

from trfl import policy_gradient_ops
from trfl import indexing_ops
from trfl import value_ops


class ProximalPolicyOptimizationLoss(collections.namedtuple(
    'ProximalPolicyOptimizationLoss', [
        'policy_gradient_loss', 
        'policy_gradient_entropy_loss', 
        'value_loss', 
        'total_loss'])):
    pass


class ProximalPolicyOptimizationAgent(agent_impl.Agent):
    """Proximal Policy Optimization (PPO) algorithm implementation.

    Computes the proximal policy optimization surrogate loss for the gradient estimation.

    Reference:
        J Schulman, et al., "Proximal Policy Optimization Algorithms".
            https://arxiv.org/abs/1707.06347
    
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
        agent = pyrl.agents.ProximalPolicyOptimizationAgent(
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
            baseline_scale=1., 
            ratio_epsilon=.2)
        trfl.update_target_variables(
            agent.behavioral_policy.trainable_variables,
            agent.policy.trainable_variables)
        ```
    """

    def __init__(self, policy, behavioral_policy, value, optimizer):
        """Creates a new ProximalPolicyOptimizationAgent.

        Args:
            policy: the target policy to optimize.
            behavioral_policy: the policy used to infer actions.
            value: the target value to optimize.
            optimizer: Instance of `tf.train.Optimizer`.
        """
        super(ProximalPolicyOptimizationAgent, self).__init__(optimizer)
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
        return ProximalPolicyOptimizationLoss(
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
                     ratio_epsilon=.2, 
                     **kwargs):
        """Computes the PPO loss.

        Args:
            states: Tensor of `[B, T, ...]` containing states.
            actions: Tensor of `[B, T, ...]` containing actions.
            rewards: Tensor of `[B, T]` containing rewards.
            weights: Tensor of shape `[B, T]` containing weights (1. or 0.).
            decay: scalar or Tensor of shape `[B, T]` containing decays/discounts.
            lambda_: scalar or Tensor of shape `[B, T]` containing generalized lambda parameter.
            entropy_scale: scalar or Tensor of shape `[B, T]` containing the entropy loss scale.
            baseline_scale: scalar or Tensor of shape `[B, T]` containing the baseline loss scale.
            ratio_epsilon: scalar or Tensor of shape `[B, T]` containing the epsilon clipping ratio.
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

        pcontinues = decay * weights
        lambda_ = lambda_ * weights

        bootstrap_values = indexing_ops.batched_index(
            baseline_values, math_ops.cast(sequence_length - 1, dtypes.int32))
        baseline_loss, td_lambda = value_ops.td_lambda(
            parray_ops.swap_time_major(baseline_values), 
            parray_ops.swap_time_major(rewards), 
            parray_ops.swap_time_major(pcontinues), 
            bootstrap_values, 
            parray_ops.swap_time_major(lambda_))

        advantages = parray_ops.swap_time_major(td_lambda.temporal_differences)
        advantages = normalization_ops.weighted_moments_normalize(advantages, weights)
        advantages = gen_array_ops.stop_gradient(advantages)

        ratio = gen_math_ops.exp(
            policy.log_prob(actions) - gen_array_ops.stop_gradient(
                behavioral_policy.log_prob(actions)))
        clipped_ratio = clip_ops.clip_by_value(ratio, 1. - ratio_epsilon, 1. + ratio_epsilon)

        self.policy_gradient_loss = -losses_impl.compute_weighted_loss(
            gen_math_ops.minimum(advantages * ratio, advantages * clipped_ratio), 
            weights=weights)

        entropy_loss = policy_gradient_ops.policy_entropy_loss(
            policy, 
            self.policy.trainable_variables,
            lambda policies: entropy_scale).loss
        self.policy_gradient_entropy_loss = losses_impl.compute_weighted_loss(
            entropy_loss,
            weights=weights)

        self.value_loss = pmath_ops.safe_divide(
            baseline_scale * math_ops.reduce_sum(baseline_loss), total_num)
        self.value_loss = gen_array_ops.check_numerics(
            self.value_loss, 'value_loss')

        self.total_loss = math_ops.add_n([
            self.value_loss,
            self.policy_gradient_loss, 
            self.policy_gradient_entropy_loss])

        return self.total_loss