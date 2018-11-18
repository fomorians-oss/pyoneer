import tensorflow as tf


def vpg_loss(policy_log_prob, advantages, entropy, weights=1., entropy_scale=.2):
    policy_loss = -tf.losses.compute_weighted_loss(advantages * policy_log_prob, weights=weights)
    policy_loss = tf.check_numerics(policy_loss, 'policy_loss')

    entropy_loss = -entropy * entropy_scale
    entropy_loss = tf.losses.compute_weighted_loss(entropy_loss, weights=weights)
    entropy_loss = tf.check_numerics(entropy_loss, 'entropy_loss')
    return policy_loss, entropy_loss
