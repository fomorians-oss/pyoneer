from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import collections
from gym import spaces

import tensorflow as tf

from pyoneer.debugging import debugging_ops
from pyoneer.rl.rollouts import unroll_ops

AgentOutput = collections.namedtuple('AgentOutput', ['action', 'log_prob'])

EnvOutput = collections.namedtuple('EnvOutput', ['state', 'reward',
                                                 'next_state', 'terminal'])


class UnrollOpsTest(tf.test.TestCase):

    def testNStepUnroll(self):
        specs = (AgentOutput(action=tf.TensorSpec([1], tf.dtypes.float32),
                             log_prob=tf.TensorSpec([], tf.dtypes.float32)),
                 EnvOutput(state=tf.TensorSpec([11, 3], tf.dtypes.int64),
                           reward=tf.TensorSpec([], tf.dtypes.float32),
                           next_state=tf.TensorSpec([11, 3], tf.dtypes.int64),
                           terminal=tf.TensorSpec([], tf.dtypes.bool)))

        initializer = debugging_ops.mock_spec(
            tf.TensorShape([1]),
            specs,
            tf.zeros)
        initial_agent_outputs, initial_env_outputs = initializer

        class Env(object):

            @tf.function
            def reset(self):
                return initial_env_outputs

            @tf.function
            def step(self, agent_outputs, env_outputs, current_step):
                return env_outputs

        class Agent(object):

            @tf.function
            def reset(self, initial_env_outputs):
                return initial_agent_outputs

            @tf.function
            def step(self, env_outputs, agent_outputs, current_step):
                return agent_outputs

        for n_step in [2, 4, 8, 12, 20, 200, 12, 8, 4, 2]:
            rollout = unroll_ops.Rollout(Env(), Agent(), n_step)

            outputs, finalizer, final_time_step = rollout()

            total_time = 0.
            for _ in range(1000):
                start = time.time()
                outputs, finalizer, final_time_step = rollout()
                total_time += (time.time() - start)
            total_time /= (1000 * n_step)
            print(total_time)

            self.assertAllEqual(n_step, final_time_step)
            tf.nest.map_structure(self.assertAllEqual, finalizer, initializer)


if __name__ == "__main__":
    tf.test.main()
