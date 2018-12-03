from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.losses import losses_impl
from tensorflow.python.eager import backprop

from pyoneer.rl.agents import agent_impl
from pyoneer.manip import array_ops as parray_ops
from pyoneer.math import normalization_ops

from trfl import policy_gradient_ops


class VanillaPolicyGradientLoss(collections.namedtuple(
    'VanillaPolicyGradientLoss', [
        'policy_gradient_loss', 'policy_gradient_entropy_loss', 'total_loss'])):
    pass


class VanillaPolicyGradientAgent(agent_impl.Agent):
    """Vanilla Policy Gradient algorithm implementation.

    Computes the policy gradient estimation:

        VPG ~= 𝔼[A(s, a)▽log(π(a|s))]
        A(s, a) ~= 𝔼[R - b(s)] 
        b(s) ~= |R - b(s)|^2

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
        return VanillaPolicyGradientLoss(
            policy_gradient_loss=self.policy_gradient_loss,
            policy_gradient_entropy_loss=self.policy_gradient_entropy_loss,
            total_loss=self.total_loss)

    def compute_loss(self, rollouts, decay=.999, entropy_scale=.2):
        returns = rollouts.discounted_returns(decay)
        self.value.fit(rollouts.states, returns)

        action_values = returns - self.value(rollouts.states, training=True)
        action_values = normalization_ops.weighted_moments_normalize(action_values, rollouts.weights)

        policy = self.policy(rollouts.states, training=True)

        log_prob = policy.log_prob(rollouts.actions)
        log_prob = parray_ops.expand_to(log_prob, ndims=3)
        policy_gradient_loss = gen_array_ops.stop_gradient(action_values) * -math_ops.reduce_sum(log_prob, axis=-1)
        self.policy_gradient_loss = losses_impl.compute_weighted_loss(
            policy_gradient_loss,
            weights=rollouts.weights)

        entropy_loss = policy_gradient_ops.policy_entropy_loss(
            policy, 
            self.policy.trainable_variables,
            lambda policies: entropy_scale).loss
        entropy_loss = math_ops.reduce_sum(parray_ops.expand_to(entropy_loss, ndims=3), axis=-1)
        self.policy_gradient_entropy_loss = losses_impl.compute_weighted_loss(
            entropy_loss,
            weights=rollouts.weights)

        self.total_loss = math_ops.add_n([
            self.policy_gradient_loss, 
            self.policy_gradient_entropy_loss])

        return self.total_loss
