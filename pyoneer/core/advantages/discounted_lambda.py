import tensorflow as tf


def discounted_lambda_advantages(rewards, values, discount=0.99, lam=0.95, weights=1.0):
    values_next = tf.concat(
        [values[:, 1:], tf.zeros_like(values[:, -1:])], axis=1)
    delta = rewards + discount * values_next - values
    advantages = tf.reverse(
        tf.transpose(
            tf.scan(
                lambda agg, cur: cur + discount * lam * agg,
                tf.transpose(tf.reverse(delta * weights, [1]), [1, 0]),
                tf.zeros_like(delta[:, -1]), 1, False), [1, 0]), [1])
    return advantages
