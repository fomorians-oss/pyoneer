import tensorflow as tf


def ppo_loss(ratio, values, returns, advantages, entropy, weights=1., epsilon=.2, entropy_scale=1.):
    clipped_ratio = tf.clip_by_value(ratio, 1. - epsilon, 1. + epsilon)
    policy_loss = -tf.losses.compute_weighted_loss(
        tf.minimum(advantages * ratio, advantages * clipped_ratio),
        weights=weights)
    policy_loss = tf.check_numerics(policy_loss, 'policy_loss')

    value_loss = .5 * tf.losses.mean_squared_error(
        predictions=values,
        labels=tf.stop_gradient(returns),
        weights=weights)
    value_loss = tf.check_numerics(value_loss, 'value_loss')

    entropy_loss = -entropy * entropy_scale
    entropy_loss = tf.losses.compute_weighted_loss(entropy_loss, weights=weights)
    entropy_loss = tf.check_numerics(entropy_loss, 'entropy_loss')
    return policy_loss, value_loss, entropy_loss
