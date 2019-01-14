from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import numpy as np

from tensorflow.python.platform import test

from pyoneer.rl.envs.batch_env_impl import BatchEnv


class BatchEnvTest(test.TestCase):
    def test_batch_env(self):
        batch_size = 2
        env = BatchEnv(
            constructor=lambda: gym.make('Pendulum-v0'), batch_size=batch_size)
        env.seed(42)
        state = env.reset()
        action = np.stack(
            [env.action_space.sample() for _ in range(batch_size)], axis=0)
        next_state, reward, done, info = env.step(action)

        self.assertTupleEqual(state.shape, (batch_size, 3))
        self.assertTupleEqual(action.shape, (batch_size, 1))
        self.assertTupleEqual(next_state.shape, (batch_size, 3))
        self.assertTupleEqual(reward.shape, (batch_size, ))
        self.assertTupleEqual(done.shape, (batch_size, ))
        self.assertAllEqual(len(info), batch_size)


if __name__ == '__main__':
    test.main()
