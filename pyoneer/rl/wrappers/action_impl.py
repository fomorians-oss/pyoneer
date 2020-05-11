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

    def unscale_action(self, action):
        # rescale from low..high to 0..1
        action = (action - self.action_space.low) / (
            self.action_space.high - self.action_space.low
        )
        # rescale from 0..1 to env.low..env.high
        action = (
            action * (self.env.action_space.high - self.env.action_space.low)
        ) + self.env.action_space.low
        return action

    def step(self, action):
        action_unscaled = self.unscale_action(action)
        observation, reward, done, info = self.env.step(action_unscaled)
        return observation, reward, done, info


class ActionProbs(gym.Wrapper):
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

    def step(self, action):
        action = np.argmax(action)
        observation, reward, done, info = self.env.step(action)
        return observation, reward, done, info


class MultiActionProbs(gym.Wrapper):
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

    def step(self, action):
        action = np.argmax(action, axis=-1)
        observation, reward, done, info = self.env.step(action)
        return observation, reward, done, info
