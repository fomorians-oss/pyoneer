from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import numpy as np


class ObservationNormalization(gym.Wrapper):
    """
    Wraps the environment to normalize observations.
    """

    def __init__(self, env, mean=None, std=None):
        super(ObservationNormalization, self).__init__(env)

        if mean is None:
            self.mean = (self.observation_space.high + self.observation_space.low) / 2
        else:
            self.mean = mean

        if std is None:
            self.std = (self.observation_space.high - self.observation_space.low) / 2
        else:
            self.std = std

        self.observation_space = self._create_observation_space()

    def _create_observation_space(self):
        low = self.mean - self.std
        high = self.mean + self.std
        return gym.spaces.Box(low, high, dtype=self.env.observation_space.dtype)

    def normalize_observation(self, observ):
        observ = (observ - self.mean) / self.std
        return observ

    def reset(self):
        return self.normalize_observation(self.env.reset())

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        observation = self.normalize_observation(observation)
        return observation, reward, done, info


class ObservationCoordinates(gym.Wrapper):
    """
    Wraps the environment to append coordinate features.

    Expects the observation space to have shape [height, width, channel]`.
    """

    def __init__(self, env):
        super(ObservationCoordinates, self).__init__(env)
        self.observation_space = self._create_observation_space()

    def _create_observation_space(self):
        observation_space = self.env.observation_space

        coord_low = np.zeros_like(observation_space.low[..., :1])
        coord_high = np.ones_like(observation_space.high[..., :1])

        coords_low = np.tile(coord_low, [1, 1, 3])
        coords_high = np.tile(coord_high, [1, 1, 3])

        low = np.concatenate([observation_space.low, coords_low], axis=-1)
        high = np.concatenate([observation_space.high, coords_high], axis=-1)

        space = gym.spaces.Box(low, high, dtype=observation_space.dtype)
        return space

    def generate_coords(self):
        observation_space = self.observation_space
        x_dim, y_dim = observation_space.shape[:2]

        xx_range = np.linspace(0.0, 1.0, num=x_dim, dtype=observation_space.dtype)
        yy_range = np.linspace(0.0, 1.0, num=y_dim, dtype=observation_space.dtype)

        xx_ones = np.ones(x_dim, dtype=observation_space.dtype)
        yy_ones = np.ones(y_dim, dtype=observation_space.dtype)

        xx_channel = np.matmul(xx_ones[..., None], xx_range[None, ...])
        yy_channel = np.matmul(yy_range[..., None], yy_ones[None, ...])
        rr_channel = np.sqrt(np.square(xx_channel) + np.square(yy_channel))
        rr_channel = rr_channel / np.amax(rr_channel)

        coords = np.stack([xx_channel, yy_channel, rr_channel], axis=-1)
        return coords

    def reset(self):
        observation = self.env.reset()
        coords = self.generate_coords()
        observation = np.concatenate([observation, coords], axis=-1)
        return observation

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        coords = self.generate_coords()
        observation = np.concatenate([observation, coords], axis=-1)
        return observation, reward, done, info
