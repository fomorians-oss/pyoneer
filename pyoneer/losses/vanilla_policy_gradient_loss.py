import tensorflow as tf

import tensorflow.losses as tfloss

def vanilla_policy_gradient_loss(log_prob,
                                 advantages,
                                 weights=1.0):
    """ Computes the Vanilla PG loss.

    Args:
      log_prob: 
      advantages:
      weights:

    Returns VanillaPolicyGradientLoss
    """

    advantages = tf.stop_gradient(advantages)

    unweighted_loss = advantages * -log_prob
    unweighted_loss = tf.check_numerics(unweighted_loss, "loss")

    loss = tfloss.compute_weighted_loss(unweighted_loss, weights=weights)
    loss = tf.check_numerics(loss, "loss")
    
    return loss




