from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_probability as tfp


class EpsilonGreedy(object):
    """
    Epsilon-greedy strategy. Samples from a policy distribution with
    `1 - epsilon` probability.

    Example:

        Decay ε every 1000 steps with a base of 0.96:

        ```
        global_step = tfe.Variable(0, trainable=False)
        initial_epsilon = 1.0 # only random samples
        strategy = EpsilonGreedyStrategy(
            tf.train.exponential_decay(
                initial_epsilon,
                global_step,
                1000,
                0.96))
        ```

    Args:
        policy: callable that returns a `tfp.distributions.Distribution`.
        epsilon: epsilon value. This can be
            a callable that takes no arguments and returns the actual
            value to use.
    """

    def __init__(self, policy, epsilon=1.0):
        self.policy = policy
        self.epsilon = epsilon

    def __call__(self, *args, **kwargs):
        policy_dist = self.policy(*args, **kwargs)

        mask_dist = tfp.distributions.Bernoulli(probs=1 - self.epsilon, dtype=tf.bool)
        mask = mask_dist.sample(policy_dist.batch_shape)

        random_logits = tf.ones_like(policy_dist.logits)
        random_dist = tfp.distributions.Categorical(logits=random_logits)

        sample = random_dist.sample()
        mode = policy_dist.mode()
        return tf.where(mask, mode, sample)


class Random(object):
    def __init__(self, policy):
        self.policy = policy

    def __call__(self, *args, **kwargs):
        policy_dist = self.policy(*args, **kwargs)
        random_logits = tf.ones_like(policy_dist.logits)
        random_dist = tfp.distributions.Categorical(logits=random_logits)
        return random_dist.sample()


class Mode(object):
    """
    Returns the mode of the policy distribution.

    Args:
        policy: callable that returns a `tfp.distributions.Distribution`.
    """

    def __init__(self, policy):
        self.policy = policy

    def __call__(self, *args, **kwargs):
        dist = self.policy(*args, **kwargs)
        return dist.mode()


class Sample(object):
    """
    Returns random samples from the policy distribution.

    Args:
        policy: callable that returns a `tfp.distributions.Distribution`.
    """

    def __init__(self, policy):
        self.policy = policy

    def __call__(self, *args, **kwargs):
        dist = self.policy(*args, **kwargs)
        return dist.sample()
