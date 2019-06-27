from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from pyoneer.math import math_ops
from pyoneer.manip import array_ops, indexing_ops


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
        returns = tf.reverse(
            tf.transpose(
                tf.scan(
                    lambda agg, cur: cur + self.discount_factor * agg,
                    tf.transpose(tf.reverse(rewards * sample_weight, [1]), [1, 0]),
                    tf.zeros_like(rewards[:, -1]),
                    1,
                    False,
                ),
                [1, 0],
            ),
            [1],
        )
        returns = returns * sample_weight
        returns = tf.debugging.check_numerics(returns, "returns")
        returns = tf.stop_gradient(returns)
        return returns


class GeneralizedAdvantages(object):
    def __init__(self, discount_factor=0.99, lambda_factor=0.95, normalize=True):
        self.discount_factor = discount_factor
        self.lambda_factor = lambda_factor
        self.normalize = normalize

    def __call__(self, rewards, values, sample_weight=1.0):
        """
        Compute generalized advantage for policy optimization. Equation 11 and 12.

        Args:
            rewards: Rewards tensor.
            discount_factor: Weighting factor for discounting.
            sample_weight: Optional sample_weight tensor.

        Returns:
            Tensor of discounted rewards.
        """
        rewards = tf.convert_to_tensor(rewards)
        values = tf.convert_to_tensor(values)
        sample_weight = tf.convert_to_tensor(sample_weight)

        sequence_lengths = tf.reduce_sum(tf.ones_like(rewards) * sample_weight, axis=1)
        last_steps = tf.cast(sequence_lengths - 1, tf.int32)
        bootstrap_values = indexing_ops.batched_index(values, last_steps)

        values_next = array_ops.shift(
            values, shift=-1, axis=1, padding_values=bootstrap_values[:, None]
        )

        deltas = rewards + self.discount_factor * values_next - values

        advantages = tf.reverse(
            tf.transpose(
                tf.scan(
                    lambda agg, cur: cur
                    + self.discount_factor * self.lambda_factor * agg,
                    tf.transpose(tf.reverse(deltas * sample_weight, [1]), [1, 0]),
                    tf.zeros_like(deltas[:, -1]),
                    1,
                    False,
                ),
                [1, 0],
            ),
            [1],
        )

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
