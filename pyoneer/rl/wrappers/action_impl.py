from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import numpy as np

from pyoneer.math import math_ops


class ActionScale(gym.Wrapper):
    """
    Wraps the environment to scale actions.
    """

    def __init__(self, env, low=-1.0, high=+1.0):
        super(ActionScale, self).__init__(env)
        assert isinstance(self.env.action_space, gym.spaces.Box)
        self.action_space = self._create_action_space(low, high)

    def _create_action_space(self, low, high):
        low = low * np.ones_like(self.env.action_space.low)
        high = high * np.ones_like(self.env.action_space.high)
        return gym.spaces.Box(low, high, dtype=self.env.action_space.dtype)

    def scale_action(self, action):
        action = (action - self.action_space.low) / (
            self.action_space.high - self.action_space.low
        )
        action = (
            action * (self.env.action_space.high - self.env.action_space.low)
        ) + self.env.action_space.low
        return action

    def step(self, action):
        action_scaled = self.scale_action(action)
        observation, reward, done, info = self.env.step(action_scaled)
        return observation, reward, done, info
