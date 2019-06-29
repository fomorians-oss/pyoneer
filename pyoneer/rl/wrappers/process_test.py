from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import tensorflow as tf

from pyoneer.rl.wrappers.process_impl import Process


class ProcessTest(tf.test.TestCase):
    def test_process(self):
        env = Process(lambda: gym.make("Pendulum-v0"))

        promise = env.seed(0)
        promise()

        promise = env.reset()
        state = promise()

        action = env.action_space.sample()
        promise = env.step(action)
        next_state, reward, done, info = promise()

        self.assertTupleEqual(state.shape, (3,))
        self.assertTupleEqual(action.shape, (1,))
        self.assertTupleEqual(next_state.shape, (3,))


if __name__ == "__main__":
    tf.test.main()
