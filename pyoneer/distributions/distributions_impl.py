from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class MultiCategorical(object):
    """
    Distribution composed of multiple distributions.

    Useful for representing `gym.spaces.MultiDiscrete`.

    Args:
        distributions: list of distributions.
    """

    def __init__(self, distributions):
        self.distributions = distributions

    def log_prob(self, value):
        values = tf.split(value, len(self.distributions), axis=-1)
        log_probs = [
            dist.log_prob(val[..., 0]) for dist, val in zip(self.distributions, values)
        ]
        return tf.math.add_n(log_probs)

    def entropy(self):
        return tf.math.add_n([dist.entropy() for dist in self.distributions])

    def sample(self):
        return tf.stack([dist.sample() for dist in self.distributions], axis=-1)

    def mode(self):
        return tf.stack([dist.mode() for dist in self.distributions], axis=-1)
