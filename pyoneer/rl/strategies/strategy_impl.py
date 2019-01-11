from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class Strategy(tf.keras.Model):
    """Base class for strategies.

    This class defines the API to add ops for directed and undirected 
    strategies using `tfp.distribution.Distribution`. You never use this 
    class directly, but instead instantiate one of its subclasses such as
    `EpsilonGreedyStrategy`.
    """

    def __init__(self, policy):
        """Creates a new Strategy.

        Args:
            policy: callable that returns a `tfp.distributions.Distribution`.
        """
        super(Strategy, self).__init__()
        self.policy = policy

    def _call_if_callable(self, param):
        """Call the function if param is callable."""
        return param() if callable(param) else param
