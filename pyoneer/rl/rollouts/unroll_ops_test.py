from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

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

        agent_specs, env_specs = specs

        initializer = debugging_ops.mock_spec(
            tf.TensorShape([1]),
            specs,
            tf.zeros)
        initial_agent_outputs, initial_env_outputs = initializer

        class Env(object):

            @property
            def output_specs(self):
                return env_specs

            @property
            def output_dtypes(self):
                return tf.nest.map_structure(
                    lambda spec: spec.dtype, self.output_specs)

            @property
            def output_shapes(self):
                return tf.nest.map_structure(
                    lambda spec: spec.shape, self.output_specs)

            @tf.function
            def reset(self):
                return initial_env_outputs

            @tf.function
            def step(self, agent_outputs, env_outputs):
                return env_outputs

        class Agent(object):

            @property
            def output_specs(self):
                return agent_specs

            @property
            def output_dtypes(self):
                return tf.nest.map_structure(
                    lambda spec: spec.dtype, self.output_specs)

            @property
            def output_shapes(self):
                return tf.nest.map_structure(
                    lambda spec: spec.shape, self.output_specs)

            @tf.function
            def reset(self, initial_env_outputs):
                return initial_agent_outputs

            @tf.function
            def step(self, env_outputs, agent_outputs):
                return agent_outputs

        for n_step in [2, 4, 8, 12, 20, 200, 12, 8, 4, 2]:
            rollout = unroll_ops.Rollout(Env(), Agent(), n_step)
            outputs, finalizer = rollout()
            tf.nest.map_structure(self.assertAllEqual, finalizer, initializer)


if __name__ == "__main__":
    tf.test.main()
