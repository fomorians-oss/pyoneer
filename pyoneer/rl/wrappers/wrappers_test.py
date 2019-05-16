from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import numpy as np

from tensorflow.python.platform import test

from pyoneer.rl.wrappers import ObservationCoordinates, ObservationNormalization


class TestEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(4, 4, 1), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(4, 4, 1), dtype=np.float32)

    def reset(self):
        return self.observation_space.sample()

    def step(self, action):
        state = self.observation_space.sample()
        reward = 0.0
        done = False
        info = {}
        return state, reward, done, info


class WrappersTest(test.TestCase):
    def test_observation_coords(self):
        env = TestEnv()
        env = ObservationCoordinates(env)

        # self.assertTupleEqual(env.observation_space.shape, (4, 4, 4))

    def test_observation_norm(self):
        env = TestEnv()
        env = ObservationNormalization(env)

        # self.assertTupleEqual(env.observation_space.shape, (4, 4, 1))


if __name__ == "__main__":
    test.main()
