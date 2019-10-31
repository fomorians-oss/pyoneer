from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import numpy as np

from pyoneer.math import math_ops


class ActionScale(gym.ActionWrapper):
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

    def action(self, action):
        action = (action - self.action_space.low) / (
            self.action_space.high - self.action_space.low
        )
        action = (
            action * (self.env.action_space.high - self.env.action_space.low)
        ) + self.env.action_space.low
        return action


class ActionProbs(gym.ActionWrapper):
    """
    Wraps the environment to support action probabilities.
    """

    def __init__(self, env):
        super(ActionProbs, self).__init__(env)
        assert isinstance(self.env.action_space, gym.spaces.Discrete)
        self.action_space = self._create_action_space()

    def _create_action_space(self):
        n = self.env.action_space.n
        low = np.zeros(shape=(n,), dtype=np.float32)
        high = np.ones(shape=(n,), dtype=np.float32)
        space = gym.spaces.Box(low, high, dtype=np.float32)
        return space

    def action(self, action):
        return np.argmax(action)


class MultiActionProbs(gym.ActionWrapper):
    """
    TODO: generalize this with gym.spaces.Tuple
    Wraps the environment to support action probabilities.
    """

    def __init__(self, env):
        super(MultiActionProbs, self).__init__(env)
        assert isinstance(self.env.action_space, gym.spaces.MultiDiscrete)
        self.action_space = self._create_action_space()

    def _create_action_space(self):
        shape = [self.env.action_space.shape[0], self.env.action_space.nvec[0]]
        low = np.zeros(shape=shape, dtype=np.float32)
        high = np.ones(shape=shape, dtype=np.float32)
        space = gym.spaces.Box(low, high, dtype=np.float32)
        return space

    def action(self, action):
        return np.argmax(action)