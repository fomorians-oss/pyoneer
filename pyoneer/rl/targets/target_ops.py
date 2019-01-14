from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from pyoneer.manip import indexing_ops
from pyoneer.math import math_ops


def discounted_rewards(rewards, discount_factor=0.99, weights=1.0):
    """
    Compute discounted rewards.

    Args:
        rewards: Rewards tensor.
        discount_factor: Weighting factor for discounting.
        weights: Optional weights tensor.

    Returns:
        Tensor of discounted rewards.
    """
    rewards = tf.convert_to_tensor(rewards)
    weights = tf.convert_to_tensor(weights)

    returns = tf.reverse(
        tf.transpose(
            tf.scan(lambda agg, cur: cur + discount_factor * agg,
                    tf.transpose(tf.reverse(rewards * weights, [1]), [1, 0]),
                    tf.zeros_like(rewards[:, -1]), 1, False), [1, 0]), [1])
    returns = returns * weights
    returns = tf.check_numerics(returns, 'returns')
    returns = tf.stop_gradient(returns)
    return returns


def generalized_advantages(rewards,
                           values,
                           discount_factor=0.99,
                           lambda_factor=0.95,
                           weights=1.0,
                           normalize=True):
    """
    Compute generalized advantage for policy optimization. Equation 11 and 12.

    Args:
        rewards: Rewards tensor.
        discount_factor: Weighting factor for discounting.
        weights: Optional weights tensor.

    Returns:
        Tensor of discounted rewards.
    """
    rewards = tf.convert_to_tensor(rewards)
    values = tf.convert_to_tensor(values)
    weights = tf.convert_to_tensor(weights)

    sequence_lengths = tf.reduce_sum(tf.ones_like(rewards) * weights, axis=1)
    last_steps = tf.cast(sequence_lengths - 1, tf.int32)
    bootstrap_values = indexing_ops.batched_index(values, last_steps)

    values_next = tf.concat([values[:, 1:], bootstrap_values[:, None]], axis=1)

    deltas = rewards + discount_factor * values_next - values

    advantages = tf.reverse(
        tf.transpose(
            tf.scan(
                lambda agg, cur: cur + discount_factor * lambda_factor * agg,
                tf.transpose(tf.reverse(deltas * weights, [1]), [1, 0]),
                tf.zeros_like(deltas[:, -1]), 1, False), [1, 0]), [1])

    if normalize:
        advantages_mean, advantages_variance = tf.nn.weighted_moments(
            advantages, axes=[0, 1], frequency_weights=weights, keep_dims=True)
        advantages_std = tf.sqrt(advantages_variance)
        advantages = math_ops.normalize(
            advantages, loc=advantages_mean, scale=advantages_std)

    advantages = advantages * weights
    advantages = tf.check_numerics(advantages, 'advantages')
    return tf.stop_gradient(advantages)
