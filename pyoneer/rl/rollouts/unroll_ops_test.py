from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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

        initializer = debugging_ops.mock_spec(tf.TensorShape([1]), specs)

        def env_step_fn(agent_outputs, env_outputs, current_step):
            return env_outputs

        def agent_step_fn(env_outputs, agent_outputs, current_step):
            return agent_outputs

        initial_time_step = 0
        for n_step in [2, 4, 8, 12]:
            outputs, finalizer, final_time_step = unroll_ops.n_step_unroll(
                n_step, initializer, agent_step_fn, env_step_fn,
                initial_time_step=initial_time_step)

            self.assertAllEqual(initial_time_step + n_step, final_time_step)
            tf.nest.map_structure(self.assertAllEqual, finalizer, initializer)

    def testNStepUnroll(self):
        specs = (AgentOutput(action=tf.TensorSpec([1], tf.dtypes.float32),
                             log_prob=tf.TensorSpec([], tf.dtypes.float32)),
                 EnvOutput(state=tf.TensorSpec([11, 3], tf.dtypes.int64),
                           reward=tf.TensorSpec([], tf.dtypes.float32),
                           next_state=tf.TensorSpec([11, 3], tf.dtypes.int64),
                           terminal=tf.TensorSpec([], tf.dtypes.bool)))

        initializer = debugging_ops.mock_spec(tf.TensorShape([1]), specs)
        initial_agent_outputs, initial_env_outputs = initializer

        class Env(object):

            def reset(self):
                return initial_env_outputs
            def step(self, agent_outputs, env_outputs, current_step):
                return env_outputs

        class Agent(object):

            def reset(self, initial_env_outputs):
                return initial_agent_outputs
            def step(self, env_outputs, agent_outputs, current_step):
                return agent_outputs

        for n_step in [2, 4, 8, 12]:
            rollout = unroll_ops.Rollout(Env(), Agent(), n_step)
            outputs, finalizer, final_time_step = rollout()
            self.assertAllEqual(n_step, final_time_step)
            tf.nest.map_structure(self.assertAllEqual, finalizer, initializer)


if __name__ == "__main__":
    tf.test.main()
