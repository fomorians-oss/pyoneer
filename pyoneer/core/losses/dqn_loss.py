import tensorflow as tf


def dqn_loss(q_values, target_q_values, returns, bootstrap_q_value, weights=1.):
    target_q_values = returns + bootstrap_q_value
    q_value_loss = tf.losses.mean_squared_error(
        predictions=q_values,
        labels=tf.stop_gradient(target_q_values),
        weights=weights)
    return q_value_loss