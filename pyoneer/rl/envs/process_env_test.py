from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym

from tensorflow.python.platform import test

from pyoneer.rl.envs.process_env_impl import ProcessEnv


class ProcessEnvTest(test.TestCase):
    def test_process_env(self):
        env = ProcessEnv(lambda: gym.make('Pendulum-v0'))
        env.seed(42)
        state = env.reset()
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)

        self.assertTupleEqual(state.shape, (3, ))
        self.assertTupleEqual(action.shape, (1, ))
        self.assertTupleEqual(next_state.shape, (3, ))


if __name__ == '__main__':
    test.main()
