import tensorflow as tf


def _n_step_discounted_returns(rewards, discount, steps, weights=1.):
    rewards = tf.convert_to_tensor(rewards)
    discount = tf.convert_to_tensor(discount)
    weights = tf.convert_to_tensor(weights)

    returns = tf.zeros_like(rewards)
    for _ in range(steps):
        returns += rewards
        rewards = discount * tf.concat(
            [rewards[:, 1:], tf.zeros_like(rewards[:, -1:])], 1)
    return returns * weights


def discounted_returns(rewards, discount, steps=None, weights=1.0):
    """Compute the n-step discounted returns.

    Args:
        rewards:
        discount:
        steps:
        weights:

    Returns:

    """
    rewards = tf.convert_to_tensor(rewards)
    discount = tf.convert_to_tensor(discount)
    weights = tf.convert_to_tensor(weights)
    if steps is None:
        return tf.reverse(
            tf.transpose(
                tf.scan(lambda agg, cur: cur + discount * agg,
                        tf.transpose(tf.reverse(rewards * weights, [1]), [1, 0]),
                        tf.zeros_like(rewards[:, -1]), 1, False), [1, 0]), [1])
    return _n_step_discounted_returns(rewards, discount, steps, weights=weights)
