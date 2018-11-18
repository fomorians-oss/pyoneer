import tensorflow as tf


def a2c_loss(policy_log_probs, values, returns, advantages, entropy, weights=1., entropy_scale=1.):
    policy_loss = -tf.losses.compute_weighted_loss(advantages * policy_log_probs, weights=weights)
    policy_loss = tf.check_numerics(policy_loss, 'policy_loss')

    value_loss = .5 * tf.losses.mean_squared_error(
        predictions=values,
        labels=tf.stop_gradient(returns))
    value_loss = tf.check_numerics(value_loss, 'value_loss')

    entropy_loss = -entropy * entropy_scale
    entropy_loss = tf.losses.compute_weighted_loss(entropy_loss, weights=weights)
    entropy_loss = tf.check_numerics(entropy_loss, 'entropy_loss')
    return policy_loss, value_loss, entropy_loss
