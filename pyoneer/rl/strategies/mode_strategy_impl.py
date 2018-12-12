from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pyoneer.rl.strategies import strategy_impl


class ModeStrategy(strategy_impl.Strategy):
    """
    Returns the mode of the policy distribution.

    Args:
        policy: callable that returns a `tfp.distributions.Distribution`.
    """

    def call(self, *args, **kwargs):
        policy = self.policy(*args, **kwargs)
        return policy.mode()
