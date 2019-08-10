from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from pyoneer.math import math_ops
from pyoneer.manip import array_ops


class DiscountedReturns(object):
    """
    Compute discounted returns.

    Args:
        rewards: Rewards tensor.
        discount_factor: Weighting factor for discounting.
        sample_weight: Optional sample_weight tensor.

    Returns:
        Tensor of discounted returns.
    """

    def __init__(self, discount_factor=0.99):
        self.discount_factor = discount_factor

    def __call__(self, rewards, sample_weight=1.0):
        def discount_step(discount_factor):
            def _discount_step(aggregate, current):
                return current + discount_factor * aggregate

            return _discount_step

        rewards_transposed = tf.transpose(rewards, [1, 0])
        initial_returns = tf.zeros_like(rewards[:, -1])

        returns_transposed = tf.scan(
            fn=discount_step(self.discount_factor),
            elems=rewards_transposed,
            initializer=initial_returns,
            back_prop=False,
            reverse=True,
        )

        returns = tf.transpose(returns_transposed, [1, 0])
        returns = tf.debugging.check_numerics(returns, "returns")
        returns = tf.stop_gradient(returns)
        return returns


class GeneralizedAdvantages(object):
    """
    Compute generalized advantage for policy optimization. Equation 11 and 12.

    Args:
        rewards: Rewards tensor.
        discount_factor: Weighting factor for discounting.
        sample_weight: Optional sample_weight tensor.

    Returns:
        Tensor of discounted rewards.
    """

    def __init__(self, discount_factor=0.99, lambda_factor=0.95, normalize=True):
        self.discount_factor = discount_factor
        self.lambda_factor = lambda_factor
        self.normalize = normalize

    def __call__(self, rewards, values, sample_weight=1.0):
        rewards = tf.convert_to_tensor(rewards)
        values = tf.convert_to_tensor(values)
        sample_weight = tf.convert_to_tensor(sample_weight)

        sample_weight = tf.ones_like(rewards) * sample_weight

        sequence_lengths = tf.reduce_sum(sample_weight, axis=1)
        last_steps = tf.cast(sequence_lengths - 1, tf.int32)

        bootstrap_values = values - values * array_ops.shift(
            sample_weight, shift=-1, axis=1
        )
        values_next = array_ops.shift(values, shift=-1, axis=1) + bootstrap_values

        deltas = (rewards + self.discount_factor * values_next - values) * sample_weight
        deltas_transposed = tf.transpose(deltas, [1, 0])

        def discount_step(discount_factor, lambda_factor):
            def _discount_step(aggregate, current):
                return current + discount_factor * lambda_factor * aggregate

            return _discount_step

        initial_advantages = tf.zeros_like(deltas[:, -1])

        advantages_transposed = tf.scan(
            fn=discount_step(self.discount_factor, self.lambda_factor),
            elems=deltas_transposed,
            initializer=initial_advantages,
            back_prop=False,
            reverse=True,
        )

        advantages = tf.transpose(advantages_transposed, [1, 0])

        if self.normalize:
            advantages_mean, advantages_variance = tf.nn.weighted_moments(
                advantages, axes=[0, 1], frequency_weights=sample_weight, keepdims=True
            )
            advantages_std = tf.sqrt(advantages_variance)
            advantages = math_ops.normalize(
                advantages,
                loc=advantages_mean,
                scale=advantages_std,
                sample_weight=sample_weight,
            )

        advantages = advantages * sample_weight
        advantages = tf.debugging.check_numerics(advantages, "advantages")
        advantages = tf.stop_gradient(advantages)
        return advantages
