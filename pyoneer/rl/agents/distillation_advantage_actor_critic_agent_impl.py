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


class DistillationAdvantageActorCriticLoss(collections.namedtuple(
    'DistillationAdvantageActorCriticLoss', [
        'distillation_loss',
        'policy_gradient_loss', 
        'policy_gradient_entropy_loss', 
        'value_loss', 
        'total_loss'])):
    pass


class DistillationAdvantageActorCriticAgent(
        advantage_actor_critic_agent_impl.MultiAdvantageActorCriticAgent):
    """Advantage Actor-Critic (A2C) with a distillation model implementation.

    Computes the actor-critic gradient estimation with a distillation 
    transformation.

    References:
        Y. Burda, et al. "Exploration by Random Network Distillation"
            https://arxiv.org/abs/1810.12894
    
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
            def call(self, inputs):
                return self.linear(inputs)

        num_actions = 2
        policy = Policy(num_actions)
        predictor_model = Model(4)
        target_model = Model(4)
        strategy = pyrl.strategies.SampleStrategy(policy)
        transformation = pyrl.transformations.DistillationBonusTransformation(
            predictor_model=predictor_model, 
            target_model=target_model)
        agent = pyrl.agents.DistillationAdvantageActorCriticAgent(
            policy=policy, 
            value=Value(2),
            predictor_model=predictor_model,
            target_model=target_model,
            optimizer=tf.train.GradientDescentOptimizer(1e-3),
            predictor_optimizer=tf.train.GradientDescentOptimizer(1e-3))
        states, actions, rewards, weights = collect_rollouts(strategy)
        rewards = transformation(states, rewards, weights, bonus_scale=1.)
        _ = agent.fit(
            states, 
            actions, 
            rewards, 
            weights, 
            decay=.999, 
            lambda_=1., 
            entropy_scale=.2, 
            baseline_scale=1.,
            distillation_scale=1.)
        ```
    """

    def __init__(self, 
                 policy, 
                 value, 
                 predictor_model, 
                 target_model, 
                 optimizer, 
                 predictor_optimizer):
        """Creates a new DistillationAdvantageActorCriticAgent.

        Args:
            policy: the target policy to optimize.
            value: the target value to optimize.
            predictor_model: A model that returns a predition 
                of the target_model returns, `predictor_model(states)`
            target_model: A model that returns anything for the 
                predictor_model to target, `target_model(states)`
            optimizer: Instance of `tf.train.Optimizer` for the 
                policy and value.
            predictor_optimizer: Instance of `tf.train.Optimizer` 
                for the predictor_model.
        """
        super(DistillationAdvantageActorCriticAgent, self).__init__(policy, value, optimizer)
        self.predictor_model = predictor_model
        self.target_model = target_model
        self.predictor_optimizer = predictor_optimizer
        self.distillation_loss = array_ops.constant(0.)

    @property
    def trainable_variables(self):
        return self.policy.trainable_variables + self.value.trainable_variables + \
            self.predictor_model.trainable_variables

    @property
    def loss(self):
        """Access recent losses computed after `compute_loss(...)` is called.

        Returns:
            a tuple containing `(distillation_loss, policy_gradient_loss, 
                policy_gradient_entropy_loss, value_loss, total_loss)`
        """
        a2c_loss = super(DistillationAdvantageActorCriticAgent, self).loss
        return DistillationAdvantageActorCriticLoss(
            **a2c_loss._asdict(),
            distillation_loss=self.distillation_loss)

    def compute_loss(self, 
                     states, 
                     actions, 
                     rewards, 
                     weights, 
                     distillation_scale=1.,
                     **kwargs):
        """Computes the A2C and distillation loss.

        Args:
            states: Tensor of `[B, T, ...]` containing states.
            actions: Tensor of `[B, T, ...]` containing actions.
            rewards: Tensor of `[B, T, 2]` containing intrinsic and 
                extrinsic rewards.
            weights: Tensor of shape `[B, T]` containing weights (1. or 0.).
            distillation_scale: scalar or Tensor of shape `[B, T]` containing 
                the distillation loss scale.
            **kwargs: keywork arguments passed to 
                `AdvantageActorCriticAgent.compute_loss(...)`.

        Returns:
            the total loss Tensor of shape [].
        """
        self.distillation_loss = distillation_scale * losses_impl.mean_squared_error(
            predictions=self.predictor_model(states),
            labels=self.target_model(states),
            weights=array_ops.expand_dims(weights, axis=-1))
        _ = super(DistillationAdvantageActorCriticAgent, self).compute_loss(
            states, actions, rewards, weights, **kwargs)
        self.total_loss += self.distillation_loss
        return self.total_loss

    def estimate_gradients(self, *args, **kwargs):
        with backprop.GradientTape(persistent=True) as tape:
            _ = self.compute_loss(*args, **kwargs)
        policy_value_variables = self.policy.trainable_variables + self.value.trainable_variables
        policy_value_gradients = tape.gradient(self.total_loss, policy_value_variables)
        predictor_gradients = tape.gradient(self.total_loss, self.predictor_model.trainable_variables)
        return (
            list(zip(policy_value_gradients, policy_value_variables)),
            list(zip(predictor_gradients, self.predictor_model.trainable_variables)))

    def fit(self, *args, **kwargs):
        policy_value_optimizer = self.optimizer
        predictor_optimizer = self.predictor_optimizer
        policy_value_grads_and_vars, predictor_grads_and_vars = self.estimate_gradients(
            *args, **kwargs)
        return control_flow_ops.group(
            policy_value_optimizer.apply_gradients(policy_value_grads_and_vars), 
            predictor_optimizer.apply_gradients(predictor_grads_and_vars))