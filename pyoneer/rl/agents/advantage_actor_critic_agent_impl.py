import collections

from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.eager import backprop

from pyoneer.rl.agents import agent_impl
from pyoneer.features import array_ops as parray_ops

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

    def compute_loss(self, rollouts, decay=.999, lambda_=1., entropy_scale=.2, baseline_scale=.5):
        policy = self.policy(parray_ops.swap_time_major(rollouts.states), training=True)
        baseline_values = self.value(rollouts.states, training=True)

        pcontinues = parray_ops.swap_time_major(decay * rollouts.weights)
        bootstrap_values = array_ops.zeros_like(baseline_values[:, 0])
        baseline_loss_td, td_lambda = value_ops.td_lambda(
            parray_ops.swap_time_major(baseline_values), 
            parray_ops.swap_time_major(rollouts.rewards), 
            pcontinues, 
            bootstrap_values, 
            lambda_)

        advantages = td_lambda.temporal_differences

        self.policy_gradient_loss = math_ops.reduce_mean(
            policy_gradient_ops.policy_gradient_loss(
                policy, 
                parray_ops.swap_time_major(rollouts.actions), 
                advantages, 
                policy_vars=self.policy.trainable_variables),
            axis=0)

        self.policy_gradient_entropy_loss = math_ops.reduce_mean(
            policy_gradient_ops.policy_entropy_loss(
                policy, 
                self.policy.trainable_variables,
                entropy_scale).loss,
            axis=0)

        self.value_loss = math_ops.reduce_mean(
            math_ops.multiply(
                parray_ops.swap_time_major(baseline_loss_td.loss), baseline_scale),
            axis=0)

        self.total_loss = math_ops.add_n([
            self.value_loss,
            self.policy_gradient_loss, 
            self.policy_gradient_entropy_loss])

        return self.total_loss
