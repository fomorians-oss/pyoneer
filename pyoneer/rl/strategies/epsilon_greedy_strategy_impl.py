from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_probability as tfp

from pyoneer.rl.strategies import strategy_impl


class EpsilonGreedyStrategy(strategy_impl.Strategy):
    """The basic ε-greedy sampling strategy."""

    def __init__(self, policy, epsilon=1.):
        """Creates a new EpsilonGreedyStrategy.

        This samples from a policy with `1 - ε` probability.

        Example:
            ```
            strategy = EpsilonGreedyStrategy(policy, 1.)
            ```

        Example: decay ε every 1000 steps with a base of 0.96:
            ```
            global_step = tfe.Variable(0, trainable=False)
            starter_epsilon = 1. # completely random start
            strategy = pyrl.EpsilonGreedyStrategy(
                tf.train.exponential_decay(
                    starter_epsilon,
                    global_step,
                    1000,
                    0.96))
            ```

        Args:
            policy: callable that returns a `tfp.distributions.Distribution`.
            initial_epsilon: the initial epsilon greedy value. This can be
                a callable that takes no arguments and returns the actual
                value to use.
        """
        super(EpsilonGreedyStrategy, self).__init__(policy)
        self.epsilon = epsilon

    def call(self, *args, **kwargs):
        policy = self.policy(*args, **kwargs)
        eps = self._call_if_callable(self.epsilon)
        mask_dist = tfp.distributions.Bernoulli(probs=1 - eps, dtype=tf.bool)
        sample_mask = mask_dist.sample(policy.batch_shape)
        sample = policy.sample()
        mode = policy.mode()
        return tf.where(sample_mask, mode, sample)
