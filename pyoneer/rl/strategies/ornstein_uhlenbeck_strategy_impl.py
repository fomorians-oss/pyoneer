from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.eager as tfe
import tensorflow_probability as tfp

from pyoneer.rl.strategies import strategy_impl


class OrnsteinUhlenbeckStrategy(strategy_impl.Strategy):
    """The Ornstein-Uhlenbeck process strategy.
    
    Implements correlated noise.
    """

    def __init__(self, policy, sigma=.3, mu=0., theta=.15):
        """Creates a new OrnsteinUhlenbeckStrategy.

        This strategy takes a continuous distribution and 
        adds time-correlated noise.

        Args:
            policy: callable that returns a `tfp.distributions.Distribution`.
            sigma: the noise standard deviation. This can be 
                a callable that takes no arguments and returns the actual 
                value to use.
            mu: the noise mean. This can be 
                a callable that takes no arguments and returns the actual 
                value to use.
            theta: the noise mean inverting scale. This can be 
                a callable that takes no arguments and returns the actual 
                value to use.
        """
        super(OrnsteinUhlenbeckStrategy, self).__init__(policy)
        self._sigma = sigma
        self._mu = mu
        self._theta = theta
        self.reset_state()

    def reset_state(self):
        """Reset the state of the strategy."""
        self._state = tf.reshape(self._call_if_callable(self._mu), [1, -1])

    def call(self, *args, **kwargs):
        policy = self.policy(*args, **kwargs)
        if isinstance(policy, tfp.distributions.Distribution):
            mode = policy.mode()
        else:
            mode = policy
        theta = self._call_if_callable(self._theta)
        mu = self._call_if_callable(self._mu)
        sigma = self._call_if_callable(self._sigma)
        sample = tfp.distributions.Normal(loc=0., scale=1.).sample(policy.batch_shape)
        if sample.shape[0] < self._state.shape[0]:
            self._state = tf.reduce_mean(self._state, axis=0)
        if sample.shape[0] > self._state.shape[0]:
            self._state = tf.tile(self._state, [sample.shape[0], 1])
        self._state += theta * (mu - self._state) + sigma * sample
        return mode + self._state
