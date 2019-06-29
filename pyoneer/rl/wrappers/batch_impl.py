from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from pyoneer.rl.wrappers.process_impl import Process


class Batch(object):
    """
    Wraps multiple `gym.Env` into a batch of environments to support
    vectorization. Uses an external processes for each environment
    when `blocking=False`.

    For simple environments, use `blocking=True` as the overhead of
    external processes can limit very large batches.

    For more complex environments, use `blocking=False` so the CPU
    computation can be offloaded.

    Note: When rendering with `mode="human"` only the first
        environment is rendered.

    Example:

    ```
    env = Batch(
        constructor=lambda: gym.make('Pendulum-v0'),
        batch_size=32,
        blocking=True)
    ```

    Args:
        constructor: Constructor which returns a `gym.Env`.
        batch_size: Number of parallel environments.
        blocking: Boolean indicating whether each call to an environment
            is blocking (default: True).
    """

    def __init__(self, constructor, batch_size, blocking=True):
        if blocking:
            self.envs = [constructor() for _ in range(batch_size)]
        else:
            self.envs = [Process(constructor) for _ in range(batch_size)]

        self.done = np.zeros(len(self.envs), dtype=np.bool)
        self.blocking = blocking

        observation_space = self.observation_space
        if not all(env.observation_space == observation_space for env in self.envs):
            raise ValueError("All environments must use the same observation space.")

        action_space = self.action_space
        if not all(env.action_space == action_space for env in self.envs):
            raise ValueError("All environments must use the same action space.")

    def __len__(self):
        return len(self.envs)

    def __getitem__(self, index):
        return self.envs[index]

    def __getattr__(self, name):
        return getattr(self.envs[0], name)

    def seed(self, seed):
        if self.blocking:
            for i, env in enumerate(self.envs):
                env.seed(seed + i)
        else:
            promises = [env.seed(seed + i) for i, env in enumerate(self.envs)]
            for promise in promises:
                promise()

    def reset(self):
        self.done[:] = False

        if self.blocking:
            states = [env.reset() for env in self.envs]
        else:
            promises = [env.reset() for env in self.envs]
            states = [promise() for promise in promises]

        state = np.stack(states, axis=0)
        return state

    def _dummy_transition(self):
        next_state = np.zeros(
            shape=self.observation_space.shape, dtype=self.observation_space.dtype
        )
        reward = 0.0
        done = True
        info = {}
        transition = (next_state, reward, done, info)
        return transition

    def step(self, actions):
        if self.blocking:
            transitions = []
            for i, env in enumerate(self.envs):
                if self.done[i]:
                    transition = self._dummy_transition()
                else:
                    transition = env.step(actions[i])
                transitions.append(transition)
        else:
            promises = []
            for i, env in enumerate(self.envs):
                if self.done[i]:
                    promise = self._dummy_transition
                else:
                    promise = env.step(actions[i])
                promises.append(promise)
            transitions = [promise() for promise in promises]

        next_states, rewards, dones, infos = zip(*transitions)
        next_state = np.stack(next_states, axis=0)
        reward = np.stack(rewards, axis=0)
        done = np.stack(dones, axis=0)
        info = tuple(infos)
        self.done = self.done | done
        return next_state, reward, done, info

    def render(self, mode="human"):
        assert (
            self.blocking or mode == "rgb_array"
        ), 'only the "rgb_array" mode is supported when `blocking=False`'

        if mode == "rgb_array":
            return np.stack([env.render(mode=mode) for env in self.envs], axis=0)
        else:
            return self.envs[0].render(mode=mode)

    def close(self):
        for env in self.envs:
            if hasattr(env, "close"):
                env.close()
