import tensorflow as tf


def differential_lambda_advantages(rewards, average_rewards, values, lam=0.95, weights=1.0):
    values_next = tf.concat(
        [values[:, 1:], tf.zeros_like(values[:, -1:])], axis=1)
    one_minus_lam = (1. - lam)
    lam_range = lam ** tf.range(0., tf.cast(rewards.shape[-1], rewards.dtype), dtype=rewards.dtype)
    advantages = -(values + (1./one_minus_lam) * average_rewards) + tf.cumsum(
        lam_range * (rewards + one_minus_lam * values_next), axis=-1, reverse=True)
    return advantages * weights


def per_step_differential_lambda_advantages(rewards, average_rewards, values, lam=0.95, weights=1.0):
    values_next = tf.concat(
        [values[:, 1:], tf.zeros_like(values[:, -1:])], axis=1)
    one_minus_lam = (1. - lam)
    lam_range = lam ** tf.range(0., tf.cast(rewards.shape[-1], rewards.dtype), dtype=rewards.dtype)
    advantages = -values + tf.cumsum(
        lam_range * (rewards - average_rewards + one_minus_lam * values_next), axis=-1, reverse=True)
    return advantages * weights
