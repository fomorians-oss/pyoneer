import tensorflow as tf


def ddpg_loss(q_values, next_q_values, actor_q_values, returns, bootstrap_q_value, weights=1.):
    target_q_values = returns + bootstrap_q_value
    critic_loss = tf.losses.mean_squared_error(
        predictions=q_values,
        labels=tf.stop_gradient(target_q_values),
        weights=weights)
    actor_loss = -tf.losses.compute_weighted_loss(actor_q_values, weights=weights)
    return critic_loss, actor_loss