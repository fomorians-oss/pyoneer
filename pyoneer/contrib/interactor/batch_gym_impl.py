from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import gym
import numpy as np
import tensorflow as tf


def wrappy(Tout, stateful=True, name=None):
    def wrapper(fn):
        def wrapped_fn(*args):
            return tf.py_func(
                fn, [*args], Tout, stateful=stateful, name=name)
        return wrapped_fn
    return wrapper


def _make_and_seed(spec):
    """Make an seed an env from the give spec and seed."""
    def wrapped(seed):
        if callable(spec):
            env = spec()
        else:
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
        cache_size = len(self.items)
        if cache_size > size:
            for item in self.items[:cache_size - size]:
                remove_fn(item)
                self.items.pop(self.items.index(item))
        elif cache_size < size:
            self.items.extend([add_fn() for _ in range(size - cache_size)])


class BatchEnv(gym.Env):

    def __init__(self, spec):
        """Create a new batched gym.Env interface from the given `str` or function spec.

        Args:
            spec: gym.Env spec str.
        """
        if callable(spec):
            self._make_and_seed = _make_and_seed(spec)
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
            episodes: scalar representing how many envs to reset.

        Returns:
            Tensor with shape `[episodes, ...observation_space.shape]`
        """
        @wrappy(tf.as_dtype(self.observation_space.dtype))
        def reset(episodes):
            self.__cache__.fix(
                lambda: self._make_and_seed(next(self.__seed__)),
                lambda item: item.close(),
                episodes)
            return np.stack([env.reset() for env in self.__cache__.items], axis=0)
        episodes = tf.convert_to_tensor(episodes)
        with tf.control_dependencies([tf.assert_greater(episodes, 0)]):
            return reset(episodes)

    def step(self, actions):
        """Step all envs with the given actions.

        Args:
            actions: numpy.ndarray with shape [envs, ...]

        Returns:
            Tuple of Tensors containing (state, reward, done, {})
        """
        @wrappy(
            [tf.as_dtype(self.observation_space.dtype), tf.float32, tf.bool])
        def step(actions):
            states = []
            rewards = []
            dones = []
            for idx, action in enumerate(actions):
                state, reward, done, _ = self.__cache__.items[idx].step(action)
                states.append(state)
                rewards.append(reward)
                dones.append(done)
            return (np.stack(states, 0), np.stack(rewards, 0), np.stack(dones, 0))

        actions = tf.convert_to_tensor(actions)
        with tf.control_dependencies(
            [tf.assert_equal(tf.shape(actions)[0], len(self.__cache__.items)),
             tf.assert_rank(actions, 1 + len(self.action_space.shape))]):
            states, rewards, dones = step(actions)
            return states, rewards, dones, {}

    def seed(self, seed=None):
        """Seed the envs generator and all envs.

        Args:
          seed: random seed.
        """
        self.__seed__ = itertools.count(seed or 1)
        for item in self.__cache__.items:
            item.seed(next(self.__seed__))

    def close(self):
        """Close the envs"""
        for env in self.__cache__.items:
            env.close()


def batch_gym_make(spec):
    """Create a new BatchEnv from the given `gym.Env` spec.

    Args:
        spec: `gym.Env` spec string or a function that returns 
            a `gym.Env`.

    Returns:
        BatchEnv
    """
    return BatchEnv(spec)
