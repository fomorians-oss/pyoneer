import collections

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.eager import backprop

from pyoneer.rl.agents import agent_impl

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

        action_values = (returns - self.value(rollouts.states, training=True)) * rollouts.weights
        policy = self.policy(rollouts.states, training=True)

        self.policy_gradient_loss = math_ops.reduce_mean(
            policy_gradient_ops.policy_gradient_loss(
                policy, 
                rollouts.actions, 
                action_values, 
                policy_vars=self.policy.trainable_variables),
            axis=0)

        self.policy_gradient_entropy_loss = math_ops.reduce_mean(
            policy_gradient_ops.policy_entropy_loss(
                policy, 
                entropy_scale_op=entropy_scale,
                policy_vars=self.policy.trainable_variables).loss,
            axis=0)

        self.total_loss = math_ops.add_n([
            self.policy_gradient_loss, 
            self.policy_gradient_entropy_loss])

        return self.total_loss
