from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import numpy as np
import tensorflow as tf

from pyoneer.rl.wrappers.action_impl import ActionScale


class TestEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(4, 4, 1), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(), dtype=np.float32
        )

    def reset(self):
        return self.observation_space.sample()

    def step(self, action):
        state = self.observation_space.sample()
        reward = 0.0
        done = False
        info = {}
        return state, reward, done, info


class ActionTest(tf.test.TestCase):
    def test_action_scale(self):
        env = TestEnv()
        env = ActionScale(env)
        self.assertTupleEqual(env.action_space.shape, ())
        self.assertAllClose(env.action_space.low, -np.ones_like(env.action_space.high))
        self.assertAllClose(env.action_space.high, np.ones_like(env.action_space.high))


if __name__ == "__main__":
    tf.test.main()
