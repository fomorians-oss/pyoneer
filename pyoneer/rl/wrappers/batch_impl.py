from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class Batch(object):
    """
    Wraps a `gym.Env` to vectorize environments.

    Example:

    ```
    env = Batch(
        constructor=lambda: gym.make('Pendulum-v0'),
        batch_size=32)
    ```

    Args:
        constructor: Constructor which returns a `gym.Env`.
        batch_size: Number of parallel environments.
    """

    def __init__(self, constructor, batch_size):
        self.envs = [constructor(batch_id) for batch_id in range(batch_size)]

    def __len__(self):
        return len(self.envs)

    def __getitem__(self, index):
        return self.envs[index]

    def __getattr__(self, name):
        return getattr(self.envs[0], name)

    def seed(self, seed=None):
        for i, env in enumerate(self.envs):
            env.seed(seed + i)

    def reset(self):
        states = [env.reset() for env in self.envs]
        state = np.stack(states, axis=0)
        return state

    def step(self, actions):
        transitions = [env.step(action) for env, action in zip(self.envs, actions)]
        next_states, rewards, dones, infos = zip(*transitions)
        next_state = np.stack(next_states, axis=0)
        reward = np.stack(rewards, axis=0)
        done = np.stack(dones, axis=0)
        info = tuple(infos)
        return next_state, reward, done, info

    def render(self, mode="human"):
        assert (
            mode == "rgb_array"
        ), 'only the "rgb_array" mode is supported by pyrl.wrappers.Batch'
        return np.stack([env.render(mode=mode) for env in self.envs], axis=0)

    def close(self):
        for env in self.envs:
            if hasattr(env, "close"):
                env.close()
