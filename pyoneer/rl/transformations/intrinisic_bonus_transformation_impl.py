from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_probability as tfp

from pyoneer.rl.transformations import transformation_impl


class IntrinsicBonusTransformation(transformation_impl.Transformation):
    """The Intrinsic Bonus.
    
    This is generic enough to implement algorithms like:
        D. Pathak, et al. "Curiosity-driven Exploration by Self-supervised Prediction"
            https://arxiv.org/abs/1705.05363
        Y. Burda, et al. "Exploration by Random Network Distillation"
            https://arxiv.org/abs/1810.12894
    """

    def __init__(self, predictor_transformation, target_transformation=None):
        """Creates a new IntrinsicBonusTransformation.

        Args:
            predictor_transformation: An arbitrary tranformation to states.
            target_transformation: An optional target transformation to states.
        """
        super(IntrinsicBonusTransformation, self).__init__()
        self.predictor_transformation = predictor_transformation
        self.target_transformation = target_transformation


class IntrinsicMotivationBonusTransformation(IntrinsicBonusTransformation):
    """Implements the Intrinsic Motivation bonus.
    
    References:
        D. Pathak, et al. "Curiosity-driven Exploration by Self-supervised Prediction"
            https://arxiv.org/abs/1705.05363
    
    Example:
        ```
        transformation = pyrl.transformations.IntrinsicMotivationBonusTransformation(
            forward_model=forward_model)
        states, next_states, actions, rewards, weights = collect_rollouts(strategy)
        rewards = transformation(
            states, next_states, actions, rewards, weights, bonus_scale=1.)
        ```
    """

    def __init__(self, forward_model):
        """Creates a new IntrinsicMotivationBonusTransformation.

        Args:
            forward_model: A forward model that returns a 
                predition of the next state, `forward_model(states, actions)`
        """     
        super(IntrinsicMotivationBonusTransformation, self).__init__(forward_model)

    def call(self, states, next_states, actions, rewards, weights, bonus_scale=1., **kwargs):
        """Transform by adding loss to rewards.

        Args:
            states: Tensor of at least 1-D containing states.
            next_states: Tensor of at least 1-D containing states[t+1].
            actions: Tensor of at least 1-D containing actions.
            rewards: Tensor of at least `dim(states) - 1`-D containing rewards.
            weights: Tensor with the same shape as `rewards` containing weights (1. or 0.).
            bonus_scale: scalar or Tensor with the same shape as `rewards` 
                containing the distillation bonus scale.
            **kwargs: keyword arguments (unused)

        Returns:
            the total rewards Tensor same shape as `rewards`.
        """
        del kwargs
        predictor_states = self.predictor_transformation(states, actions)
        target_states = next_states
        prediction_bonus = tf.reduce_sum(
            bonus_scale * tf.square(predictor_states - target_states) * tf.expand_dims(weights, axis=-1), 
            axis=-1)
        rewards += prediction_bonus
        return rewards


class DistillationBonusTransformation(IntrinsicBonusTransformation):
    """Implements the Distillation bonus.

    References:
        Y. Burda, et al. "Exploration by Random Network Distillation"
            https://arxiv.org/abs/1810.12894
    
    Example:
        ```
        transformation = pyrl.transformations.DistillationBonusTransformation(
            predictor_model=predictor_model, 
            target_model=target_model)
        states, actions, rewards, weights = collect_rollouts(strategy)
        rewards = transformation(states, rewards, weights, bonus_scale=1.)
        ```
    """

    def __init__(self, predictor_model, target_model):
        """Creates a new DistillationBonusTransformation.

        Args:
            predictor_model: A model that returns a predition 
                of the target_model returns, `predictor_model(states)`
            target_model: A model that returns anything for the 
                predictor_model to target, `target_model(states)`
        """     
        super(DistillationBonusTransformation, self).__init__(
            predictor_model, target_model)

    def call(self, states, rewards, weights, bonus_scale=1., **kwargs):
        """Transform by stacking distillation loss on rewards.

        Args:
            states: Tensor of at least 1-D containing states.
            rewards: Tensor of at least `dim(states) - 1`-D containing rewards.
            weights: Tensor with the same shape as `rewards` containing weights (1. or 0.).
            bonus_scale: scalar or Tensor with the same shape as `rewards` 
                containing the distillation bonus scale.
            **kwargs: keyword arguments (unused)

        Returns:
            the stacked rewards Tensor of shape [..., 2].
        """
        del kwargs
        predictor_states = self.predictor_transformation(states)
        target_states = self.target_transformation(states)
        prediction_bonus = tf.reduce_sum(
            bonus_scale * tf.square(predictor_states - tf.stop_gradient(target_states)) * tf.expand_dims(
                weights, axis=-1), 
            axis=-1)
        rewards = tf.stack([prediction_bonus, rewards], axis=-1)
        return rewards