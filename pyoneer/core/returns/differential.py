import tensorflow as tf


def _n_step_differential_returns(rewards, average_rewards, steps, weights=1.):
    rewards = tf.convert_to_tensor(rewards)
    average_rewards = tf.tile([average_rewards], [rewards.shape[-1]])
    weights = tf.convert_to_tensor(weights)

    returns = tf.zeros_like(rewards)
    for i in range(steps):
        returns += (rewards - average_rewards)
        rewards = tf.concat([rewards[:, 1:], tf.zeros_like(rewards[:, -1:])], 1)
        average_rewards = tf.concat([average_rewards[1:], tf.zeros_like(average_rewards[-1:])], 0)
    return returns * weights


def differential_returns(rewards, average_rewards, steps=None, weights=1.0):
    """Compute the n-step differential returns.

    Args:
        rewards:
        average_rewards:
        steps:
        weights:

    Returns:

    """
    rewards = tf.convert_to_tensor(rewards)
    weights = tf.convert_to_tensor(weights)

    if steps is None:
        cumulative_rewards = tf.cumsum(rewards, axis=-1, reverse=True)
        cumulative_average_rewards = -tf.range(
            -tf.cast(tf.shape(rewards)[-1], rewards.dtype), 0.,
            dtype=rewards.dtype) * average_rewards
        return weights * (cumulative_rewards - cumulative_average_rewards)
    return _n_step_differential_returns(rewards, average_rewards, steps + 1, weights=weights)


def _n_step_per_step_differential_returns(rewards, average_rewards, steps, weights=1.):
    rewards = tf.convert_to_tensor(rewards)
    weights = tf.convert_to_tensor(weights)

    returns = tf.zeros_like(rewards)
    for i in range(steps):
        returns += (rewards - average_rewards)
        rewards = tf.concat([rewards[:, 1:], tf.zeros_like(rewards[:, -1:])], 1)
        average_rewards = tf.concat([average_rewards[1:], tf.zeros_like(average_rewards[-1:])], 0)
    return returns * weights


def per_step_differential_returns(rewards, average_rewards, steps=None, weights=1.0):
    """Compute the n-step differential returns.

    Args:
        rewards:
        average_rewards:
        steps:
        weights:

    Returns:

    """
    rewards = tf.convert_to_tensor(rewards)
    weights = tf.convert_to_tensor(weights)

    if steps is None:
        cumulative_rewards = tf.cumsum(rewards, axis=-1, reverse=True)
        cumulative_average_rewards = tf.cumsum(average_rewards, axis=-1, reverse=True)
        return weights * (cumulative_rewards - cumulative_average_rewards)
    return _n_step_per_step_differential_returns(rewards, average_rewards, steps + 1, weights=weights)
