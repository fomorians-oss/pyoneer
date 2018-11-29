import collections

from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.eager import backprop
from tensorflow.python.ops.losses import losses_impl

from pyoneer.rl.agents import agent_impl
from pyoneer.features import array_ops as parray_ops

from trfl import policy_gradient_ops
from trfl import value_ops


class ProximalPolicyOptimizationLoss(collections.namedtuple(
    'ProximalPolicyOptimizationLoss', [
        'policy_gradient_loss', 'policy_gradient_entropy_loss', 'value_loss', 'total_loss'])):
    pass


class ProximalPolicyOptimizationAgent(agent_impl.Agent):
    """Proximal Policy Optimization (PPO) algorithm implementation.

    Computes the proximal policy optimization surrogate loss for the gradient estimation:
    """

    def __init__(self, policy, behavioral_policy, value, optimizer):
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
        return ProximalPolicyOptimizationLoss(
            policy_gradient_loss=self.policy_gradient_loss,
            policy_gradient_entropy_loss=self.policy_gradient_entropy_loss,
            value_loss=self.value_loss,
            total_loss=self.total_loss)

    def compute_loss(self, rollouts, decay=.999, lambda_=1., entropy_scale=.2, baseline_scale=.5, ratio_epsilon=.2):
        policy = self.policy(rollouts.states, training=True)
        behavioral_policy = self.behavioral_policy(rollouts.states)
        baseline_values = self.value(rollouts.states, training=True)

        pcontinues = decay * rollouts.weights
        lambda_ = lambda_ * rollouts.weights
        bootstrap_values = array_ops.zeros_like(baseline_values[:, 0])
        baseline_loss, td_lambda = value_ops.td_lambda(
            parray_ops.swap_time_major(baseline_values), 
            parray_ops.swap_time_major(rollouts.rewards), 
            parray_ops.swap_time_major(pcontinues), 
            bootstrap_values, 
            parray_ops.swap_time_major(lambda_))

        advantages = parray_ops.swap_time_major(td_lambda.temporal_differences)

        ratio = gen_math_ops.exp(gen_math_ops.log(behavioral_policy) - gen_math_ops.log(policy))
        clipped_ratio = clip_ops.clip_by_value(ratio, 1. - ratio_epsilon, 1. + ratio_epsilon)
        
        self.policy_gradient_loss = -losses_impl.compute_weighted_loss(
            gen_math_ops.minimum(advantages * ratio, advantages * clipped_ratio), 
            weights=rollouts.weights)

        self.policy_gradient_entropy_loss = math_ops.reduce_mean(
            policy_gradient_ops.policy_entropy_loss(
                policy, 
                self.policy.trainable_variables,
                entropy_scale).loss, 
            axis=0)

        self.value_loss = math_ops.reduce_mean(
            math_ops.multiply(baseline_loss, baseline_scale), 
            axis=0)

        self.total_loss = math_ops.add_n([
            self.value_loss,
            self.policy_gradient_loss, 
            self.policy_gradient_entropy_loss])

        return self.total_loss