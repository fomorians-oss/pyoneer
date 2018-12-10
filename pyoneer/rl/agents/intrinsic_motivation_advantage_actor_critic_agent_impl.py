from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops.losses import losses_impl
from tensorflow.python.eager import backprop

from pyoneer.manip import array_ops as parray_ops
from pyoneer.math import normalization_ops
from pyoneer.math import math_ops as pmath_ops
from pyoneer.rl.agents import advantage_actor_critic_agent_impl

from trfl import policy_gradient_ops
from trfl import value_ops


class IntrinsicMotivationAdvantageActorCriticLoss(collections.namedtuple(
    'IntrinsicMotivationAdvantageActorCriticLoss', [
        'motivation_loss',
        'policy_gradient_loss', 
        'policy_gradient_entropy_loss', 
        'value_loss', 
        'total_loss'])):
    pass


class IntrinsicMotivationAdvantageActorCriticAgent(
        advantage_actor_critic_agent_impl.AdvantageActorCriticAgent):
    """Advantage Actor-Critic (A2C) with an intrinsic motivation model implementation.

    Computes the actor-critic gradient estimation with a distillation 
    transformation.

    References:
        D. Pathak, et al. "Curiosity-driven Exploration by Self-supervised Prediction"
            https://arxiv.org/abs/1705.05363
    
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

        class Model(tf.keras.Model):
            def __init__(self, num_units):
                super(Model, self).__init__()
                self.linear = tf.layers.Dense(num_units)
            def call(self, states, actions):
                return self.linear(tf.concat([states, actions], axis=-1))

        num_actions = 2
        policy = Policy(num_actions)
        forward_model = Model(5)
        strategy = pyrl.strategies.SampleStrategy(policy)
        transformation = pyrl.transformations.IntrinsicMotivationBonusTransformation(
            forward_model=forward_model)
        agent = pyrl.agents.IntrinsicMotivationAdvantageActorCriticAgent(
            policy=policy, 
            value=Value(1),
            forward_model=forward_model,
            optimizer=tf.train.GradientDescentOptimizer(1e-3),
            forward_optimizer=tf.train.GradientDescentOptimizer(1e-3))
        states, next_states, actions, rewards, weights = collect_rollouts(strategy)
        rewards = transformation(
            states, next_states, actions, rewards, weights, bonus_scale=1.)
        _ = agent.fit(
            states, 
            actions, 
            rewards, 
            weights, 
            decay=.999, 
            lambda_=1., 
            entropy_scale=.2, 
            baseline_scale=1.,
            motivation_scale=1.)
        ```
    """

    def __init__(self, 
                 policy, 
                 value, 
                 forward_model, 
                 optimizer, 
                 forward_optimizer):
        """Creates a new IntrinsicMotivationAdvantageActorCriticAgent.

        Args:
            policy: the target policy to optimize.
            value: the target value to optimize.
            forward_model: A model that returns a predition 
                of the target_model returns, `forward_model(states, actions)`
            optimizer: Instance of `tf.train.Optimizer` for the 
                policy and value.
            forward_optimizer: Instance of `tf.train.Optimizer` 
                for the forward_model.
        """
        super(IntrinsicMotivationAdvantageActorCriticAgent, self).__init__(policy, value, optimizer)
        self.forward_model = forward_model
        self.forward_optimizer = forward_optimizer
        self.motivation_loss = array_ops.constant(0.)

    @property
    def trainable_variables(self):
        return self.policy.trainable_variables + self.value.trainable_variables + \
            self.forward_model.trainable_variables

    @property
    def loss(self):
        """Access recent losses computed after `compute_loss(...)` is called.

        Returns:
            a tuple containing `(motivation_loss, policy_gradient_loss, 
                policy_gradient_entropy_loss, value_loss, total_loss)`
        """
        a2c_loss = super(IntrinsicMotivationAdvantageActorCriticAgent, self).loss
        return IntrinsicMotivationAdvantageActorCriticLoss(
            **a2c_loss._asdict(),
            motivation_loss=self.motivation_loss)

    def compute_loss(self, 
                     states, 
                     next_states,
                     actions, 
                     rewards, 
                     weights, 
                     motivation_scale=1.,
                     **kwargs):
        """Computes the A2C and intrinsic motivation loss.

        Args:
            states: Tensor of `[B, T, ...]` containing states.
            next_states: Tensor of `[B, T, ...]` containing states[t+1].
            actions: Tensor of `[B, T, ...]` containing actions.
            rewards: Tensor of `[B, T, 2]` containing intrinsic and 
                extrinsic rewards.
            weights: Tensor of shape `[B, T]` containing weights (1. or 0.).
            motivation_scale: scalar or Tensor of shape `[B, T]` containing 
                the motivation loss scale.
            **kwargs: keywork arguments passed to 
                `AdvantageActorCriticAgent.compute_loss(...)`.

        Returns:
            the total loss Tensor of shape [].
        """
        self.motivation_loss = motivation_scale * losses_impl.mean_squared_error(
            predictions=self.forward_model(states, actions),
            labels=next_states,
            weights=array_ops.expand_dims(weights, axis=-1))
        _ = super(IntrinsicMotivationAdvantageActorCriticAgent, self).compute_loss(
            states, actions, rewards, weights, **kwargs)
        self.total_loss += self.motivation_loss
        return self.total_loss

    def estimate_gradients(self, *args, **kwargs):
        with backprop.GradientTape(persistent=True) as tape:
            _ = self.compute_loss(*args, **kwargs)
        policy_value_variables = self.policy.trainable_variables + self.value.trainable_variables
        policy_value_gradients = tape.gradient(self.total_loss, policy_value_variables)
        forward_gradients = tape.gradient(self.total_loss, self.forward_model.trainable_variables)
        return (
            list(zip(policy_value_gradients, policy_value_variables)),
            list(zip(forward_gradients, self.forward_model.trainable_variables)))

    def fit(self, *args, **kwargs):
        policy_value_optimizer = self.optimizer
        forward_optimizer = self.forward_optimizer
        policy_value_grads_and_vars, forward_grads_and_vars = self.estimate_gradients(
            *args, **kwargs)
        return control_flow_ops.group(
            policy_value_optimizer.apply_gradients(policy_value_grads_and_vars), 
            forward_optimizer.apply_gradients(forward_grads_and_vars))