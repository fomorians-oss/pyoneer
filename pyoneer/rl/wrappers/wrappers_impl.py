import gym
import numpy as np


class ObservationNormalization:
    def __init__(self, env):
        assert isinstance(env.observation_space, gym.spaces.Box)
        self.env = env

    def __getattr__(self, name):
        return getattr(self.env, name)

    @property
    def observation_space(self):
        observation_space = self.env.observation_space
        low = -np.ones(shape=observation_space.shape, dtype=observation_space.dtype)
        high = np.ones(shape=observation_space.shape, dtype=observation_space.dtype)
        return gym.spaces.Box(low, high, dtype=observation_space.dtype.dtype)

    def normalize_observation(self, observ):
        low = self.env.observation_space.low
        high = self.env.observation_space.high
        observ = 2 * (observ - low) / (high - low) - 1
        return observ

    def reset(self):
        return self.normalize_observation(self.env.reset())

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        observation = self.normalize_observation(observation)
        return observation, reward, done, info


class ObservationCoordinates:
    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        return getattr(self.env, name)

    @property
    def observation_space(self):
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
        x_dim, y_dim = self.observation_space.shape[:2]

        xx_range = np.linspace(0.0, 1.0, x_dim)
        yy_range = np.linspace(0.0, 1.0, y_dim)

        xx_ones = np.ones(x_dim, dtype=self.observation_space.dtype)
        yy_ones = np.ones(y_dim, dtype=self.observation_space.dtype)

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
