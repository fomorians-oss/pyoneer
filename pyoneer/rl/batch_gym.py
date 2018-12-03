from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import gym
import numpy as np


def _make_and_seed(spec):
    def wrapped(seed):
        """Make an seed an env from the give spec and seed.

        Args:
          spec: gym.Env spec str.
          seed: Random seed.

        Returns:
          gym.Env
        """
        env = gym.make(spec)
        env.seed(seed)
        return env
    return wrapped


class _Cache(object):

    def __init__(self, items):
        """Create a new cache.

        Args:
          items: list of elements in the cache.
        """
        self.items = items

    def fix(self, add_fn, remove_fn, size):
        """Fix the cache to the specified size.

        Args:
          add_fn: called when a new item is added to the cache.
          remove_fn: called when a new item is removed from the cache.
          size: desired cache size.
        """
        if len(self.items) > size:
            for item in self.items[size:]:
                remove_fn(item)
            self.items = self.items[:-size]
        elif len(self.items) < size:
            self.items.extend([add_fn() for _ in range(size - 1)])


class BatchEnv(gym.Env):

    def __init__(self, spec):
        """Create a new batched gym.Env interface from the given `str` or function spec.

        Args:
          spec: gym.Env spec str.
        """
        if callable(spec):
            self._make_and_seed = spec
            env = spec()
        else:
            self._make_and_seed = _make_and_seed(spec)
            env = gym.make(spec)

        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self.__seed__ = itertools.count(0)
        self.__cache__ = _Cache([env])

    @property
    def unwrapped(self):
        return getattr(self.__cache__.items[0], 'unwrapped')

    def __getattr__(self, item):
        return getattr(self.__cache__.items[0], item)

    def reset(self, episodes=1):
        """Fix size of the cache and reset envs

        Args:
          episodes: number of envs to reset.

        Returns:
          state
        """
        assert episodes > 0
        if episodes != len(self.__cache__.items):
            self.__cache__.fix(
                lambda: self._make_and_seed(next(self.__seed__)),
                lambda item: item.close(),
                episodes)
        return np.stack([env.reset() for env in self.__cache__.items], 0)

    def step(self, actions):
        """Step all environments from the given actions.

        Args:
          actions: numpy.ndarray with shape [envs, ...]

        Returns:
          state, reward, done, info
        """
        assert actions.shape[0] == len(self.__cache__.items)
        states = []
        rewards = []
        dones = []
        infos = []

        for idx, action in enumerate(actions):
            state, reward, done, info = self.__cache__.items[idx].step(action)
            states.append(state)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)

        return (
            np.stack(states, 0),
            np.stack(rewards, 0),
            np.stack(dones, 0),
            infos)

    def seed(self, seed=None):
        """Seed the envs.

        Args:
          seed: random seed.
        """
        self.__seed__ = itertools.count(seed or 0)
        for item in self.__cache__.items:
            item.seed(next(self.__seed__))

    def close(self):
        """Close the cached envs"""
        for env in self.__cache__.items:
            env.close()


def batch_gym_make(spec):
    """Create a new batched gym.Env interface from the given gym.Env spec.

    Args:
      spec: gym.Env spec.

    Returns:
      batched gym.Env interface.
    """
    return BatchEnv(spec)
