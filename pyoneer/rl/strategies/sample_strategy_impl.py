from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class SampleStrategy:
    """
    Returns random samples from the policy distribution.

    Args:
        policy: callable that returns a `tfp.distributions.Distribution`.
    """

    def __init__(self, policy):
        self.policy = policy

    def __call__(self, *args, **kwargs):
        policy = self.policy(*args, **kwargs)
        return policy.sample()
