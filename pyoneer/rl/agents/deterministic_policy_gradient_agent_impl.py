import collections

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.eager import backprop
from tensorflow.python.ops.losses import losses_impl
from tensorflow.python.training import optimizer

from pyoneer.rl.agents import agent_impl
from pyoneer.features import array_ops as parray_ops

from trfl import policy_gradient_ops
from trfl import value_ops


class DeterministicPolicyGradientLoss(collections.namedtuple(
    'DeterministicPolicyGradient', [
        'policy_gradient_loss', 'policy_gradient_entropy_loss', 'value_loss', 'total_loss'])):
    pass


class DeterministicPolicyGradientAgent(agent_impl.Agent):
    """Deterministic Policy Gradient (DPG) algorithm implementation.

    Computes the deterministic policy gradient estimation:
    """

    def __init__(self, policy, behavioral_policy, value, optimizer):
        assert isinstance(optimizer, tuple)
        super(DeterministicPolicyGradientAgent, self).__init__(optimizer)

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
        return DeterministicPolicyGradientLoss(
            policy_gradient_loss=self.policy_gradient_loss,
            policy_gradient_entropy_loss=self.policy_gradient_entropy_loss,
            value_loss=self.value_loss,
            total_loss=self.total_loss)

    def compute_loss(self, rollouts, delay=.999, lambda_=1., entropy_scale=.2):
        sequence_length = math_ops.reduce_sum(rollouts.weights, axis=1)
        mask = array_ops.sequence_mask(
            gen_math_ops.maximum(sequence_length - 1, 0), 
            maxlen=rollouts.states.shape[1], 
            dtype=dtypes.float32)

        policy = self.policy(rollouts.states, training=True)
        behavioral_policy = self.behavioral_policy(rollouts.next_states)

        bootstrap_state = array_ops.expand_dims(
            array_ops.gather(rollouts.next_states, sequence_length), 
            axis=1)
        bootstrap_action = array_ops.expand_dims(
            array_ops.gather(behavioral_policy.mode(), sequence_length), 
            axis=1)
        bootstrap_value = array_ops.squeeze(
            self.value(bootstrap_state, bootstrap_action), 
            axis=1)

        action_values = self.value(rollouts.states, policy.mode(), training=True) * mask
        self.policy_gradient_loss = losses_impl.compute_weighted_loss(
            -action_values, weights=rollouts.weights)

        policy_gradient_entropy_loss_output = policy_gradient_ops.policy_entropy_loss(
            policy, 
            self.policy.trainable_variables,
            entropy_scale)
        self.policy_gradient_entropy_loss = losses_impl.compute_weighted_loss(
            policy_gradient_entropy_loss_output.loss, weights=rollouts.weights)

        pcontinues = parray_ops.swap_time_major(delay * rollouts.weights)
        self.value_loss = math_ops.reduce_mean(
            value_ops.td_lambda(
                action_values, 
                rollouts.rewards,
                pcontinues,
                gen_array_ops.stop_gradient(bootstrap_value),
                lambda_=lambda_).loss,
            axis=0)

        self.total_loss = math_ops.add_n([
            self.value_loss,
            self.policy_gradient_loss, 
            self.policy_gradient_entropy_loss])

        return self.total_loss

    def estimate_gradients(self, rollouts, **kwargs):
        with backprop.GradientTape(persistent=True) as tape:
            _ = self.compute_loss(rollouts, **kwargs)
        policy_gradients = tape.gradient(self.total_loss, self.policy.trainable_variables)
        value_gradients = tape.gradient(self.total_loss, self.value.trainable_variables)
        return policy_gradients, value_gradients