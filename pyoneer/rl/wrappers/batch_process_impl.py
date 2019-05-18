from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import multiprocessing

from pyoneer.rl.wrappers.process_impl import Process


class BatchProcess(object):
    """
    Wraps a `gym.Env` to host a batch of environments in external processes.

    Example:

    ```
    env = BatchProcess(
        constructor=lambda: gym.make('Pendulum-v0'),
        batch_size=32,
        blocking=False)
    ```

    Args:
        constructor: Constructor which returns a `gym.Env`.
        batch_size: Number of parallel environments.
        blocking: Boolean indicating whether each call to the environment is
            blocking.
    """

    def __init__(self, constructor, batch_size=None, blocking=False):
        if batch_size is None:
            batch_size = multiprocessing.cpu_count()

        self.envs = [Process(constructor) for _ in range(batch_size)]
        self.blocking = blocking

        observation_space = self.observation_space
        if not all(env.observation_space == observation_space for env in self.envs):
            raise ValueError("All environments must use the same observation space.")

        action_space = self.action_space
        if not all(env.action_space == action_space for env in self.envs):
            raise ValueError("All environments must use the same observation space.")

    def __len__(self):
        return len(self.envs)

    def __getitem__(self, index):
        return self.envs[index]

    def __getattr__(self, name):
        return getattr(self.envs[0], name)

    def seed(self, seed=None):
        if self.blocking:
            for i, env in enumerate(self.envs):
                env.seed(seed + i)
        else:
            promises = [env.seed(seed + i) for i, env in enumerate(self.envs)]
            for promise in promises:
                promise()

    def reset(self):
        if self.blocking:
            states = [env.reset() for env in self.envs]
        else:
            promises = [env.reset() for env in self.envs]
            states = [promise() for promise in promises]

        state = np.stack(states, axis=0)
        return state

    def step(self, actions):
        if self.blocking:
            transitions = [env.step(action) for env, action in zip(self.envs, actions)]
        else:
            promises = [env.step(action) for env, action in zip(self.envs, actions)]
            transitions = [promise() for promise in promises]

        next_states, rewards, dones, infos = zip(*transitions)
        next_state = np.stack(next_states, axis=0)
        reward = np.stack(rewards, axis=0)
        done = np.stack(dones, axis=0)
        info = tuple(infos)
        return next_state, reward, done, info

    def close(self):
        for env in self.envs:
            if hasattr(env, "close"):
                env.close()
