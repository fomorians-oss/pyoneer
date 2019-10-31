from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
import collections
import gym
import tensorflow as tf

from pyoneer.debugging import debugging_ops
from pyoneer.rl.wrappers import batch_impl
from pyoneer.rl.rollouts import unroll_ops
from pyoneer.rl.rollouts import gym_ops

AgentOutput = collections.namedtuple('AgentOutput', 'action')


class EnvTest(tf.test.TestCase):

    def testEnvResetStepSeeding(self):
        zero = tf.zeros([1], tf.dtypes.int64)

        def make_env():
            env = gym.make('CartPole-v0')
            return env

        gym_env = make_env()
        module_env = gym_ops.Env(gym_env)

        module_env.seed(42)
        dummy_env_reset_output = module_env.reset()
        dummy_env_step_output0 = module_env.step(
            AgentOutput(zero), dummy_env_reset_output)
        dummy_env_step_output1 = module_env.step(
            AgentOutput(zero), dummy_env_step_output0)

        module_env.seed(42)
        env_reset_output = module_env.reset()
        tf.nest.map_structure(
            self.assertAllEqual, (dummy_env_reset_output,),
            (env_reset_output,))
        env_step_output0 = module_env.step(
            AgentOutput(zero), env_reset_output)
        tf.nest.map_structure(
            self.assertAllEqual, (dummy_env_step_output0,),
            (env_step_output0,))
        env_step_output1 = module_env.step(
            AgentOutput(zero), env_step_output0)
        tf.nest.map_structure(
            self.assertAllEqual, (dummy_env_step_output1,),
            (env_step_output1,))

    def testEnvResetStepSeedingBatch(self):
        batch_size = 10
        zero = tf.zeros([batch_size], tf.dtypes.int64)

        def make_env():
            env = gym.make('CartPole-v0')
            return env

        gym_env = batch_impl.Batch(make_env, batch_size)
        module_env = gym_ops.Env(gym_env)

        module_env.seed(42)
        dummy_env_reset_output = module_env.reset()
        dummy_env_step_output0 = module_env.step(
            AgentOutput(zero), dummy_env_reset_output)
        dummy_env_step_output1 = module_env.step(
            AgentOutput(zero), dummy_env_step_output0)

        module_env.seed(42)
        env_reset_output = module_env.reset()
        tf.nest.map_structure(
            self.assertAllEqual, (dummy_env_reset_output,),
            (env_reset_output,))
        env_step_output0 = module_env.step(
            AgentOutput(zero), env_reset_output)
        tf.nest.map_structure(
            self.assertAllEqual, (dummy_env_step_output0,),
            (env_step_output0,))
        env_step_output1 = module_env.step(
            AgentOutput(zero), env_step_output0)
        tf.nest.map_structure(
            self.assertAllEqual, (dummy_env_step_output1,),
            (env_step_output1,))

    def testRolloutEnv(self):
        batch_size = 1
        zero = tf.zeros([batch_size, 1], tf.dtypes.float32)

        def make_env():
            env = gym.make('Pendulum-v0')
            return env

        gym_env = make_env()
        module_env = gym_ops.Env(gym_env)

        class Agent(object):

            @property
            def output_specs(self):
                return tf.TensorSpec([1], tf.dtypes.float32)

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
                return AgentOutput(zero)

            @tf.function
            def step(self, env_outputs, agent_outputs):
                return AgentOutput(zero)

        initial_time_step = 0
        for time_major in [True, False]:
            for random_seed, n_step in zip([42, 47, 1, 100], [2, 4, 8, 12]):
                # Simulate unroll.
                module_env.seed(random_seed)
                expected_env_outputs_initializer = module_env.reset()
                env_outputs = expected_env_outputs_initializer
                expected_env_outputs_finalizer = expected_env_outputs_initializer
                expected_env_outputs = []

                for step in range(1, n_step + 1):
                    if not expected_env_outputs_finalizer.terminal:
                        expected_env_outputs_finalizer = module_env.step(
                            AgentOutput(zero), expected_env_outputs_finalizer)
                        expected_env_outputs.append(expected_env_outputs_finalizer)
                    else:
                        # Terminal state.
                        # Simulate padding.
                        env_outputs_padding = debugging_ops.mock_spec(
                            tf.TensorShape([batch_size]),
                            module_env.output_specs,
                            tf.zeros)
                        for _ in range(step, n_step + 1):
                            expected_env_outputs.append(env_outputs_padding)
                        step -= 1
                        break

                expected_env_outputs = tf.nest.map_structure(
                    lambda *args: tf.stack(args, axis=int(not time_major)),
                    *expected_env_outputs)

                # Unroll.
                module_env.seed(random_seed)
                rollout = unroll_ops.Rollout(module_env, Agent(), n_step, time_major=time_major)

                ((_, env_outputs), (_, env_outputs_finalizer)) = rollout()

                tf.nest.map_structure(self.assertAllEqual, env_outputs,
                                    expected_env_outputs)
                tf.nest.map_structure(self.assertAllEqual, env_outputs_finalizer,
                                    expected_env_outputs_finalizer)

    def testRolloutEnvBatch(self):
        batch_size = 1
        zero = tf.zeros([batch_size, 1], tf.dtypes.float32)

        def make_env():
            env = gym.make('Pendulum-v0')
            return env

        gym_env = batch_impl.Batch(make_env, batch_size)
        module_env = gym_ops.Env(gym_env)

        class Agent(object):

            @property
            def output_specs(self):
                return tf.TensorSpec([1], tf.dtypes.float32)

            @property
            def output_dtypes(self):
                return tf.nest.map_structure(
                    lambda spec: spec.dtype, self.output_specs)

            @property
            def output_shapes(self):
                return tf.nest.map_structure(
                    lambda spec: spec.shape, self.output_specs)

            def reset(self, initial_env_outputs):
                return AgentOutput(zero)

            def step(self, env_outputs, agent_outputs):
                return AgentOutput(zero)

        initial_time_step = 0
        for time_major in [True, False]:
            for random_seed, n_step in zip([42, 47, 1, 100], [2, 4, 8, 12]):
                # Simulate unroll.
                module_env.seed(random_seed)
                expected_env_outputs_initializer = module_env.reset()
                env_outputs = expected_env_outputs_initializer
                expected_env_outputs_finalizer = expected_env_outputs_initializer
                expected_env_outputs = []

                for step in range(1, n_step + 1):
                    if not expected_env_outputs_finalizer.terminal:
                        expected_env_outputs_finalizer = module_env.step(
                            AgentOutput(zero), expected_env_outputs_finalizer)
                        expected_env_outputs.append(expected_env_outputs_finalizer)
                    else:
                        # Terminal state.
                        # Simulate padding.
                        env_outputs_padding = debugging_ops.mock_spec(
                            tf.TensorShape([batch_size]),
                            module_env.output_specs,
                            tf.zeros)
                        for _ in range(step, n_step + 1):
                            expected_env_outputs.append(env_outputs_padding)
                        step -= 1
                        break

                expected_env_outputs = tf.nest.map_structure(
                    lambda *args: tf.stack(args, axis=int(not time_major)),
                    *expected_env_outputs)

                # Unroll.
                module_env.seed(random_seed)
                rollout = unroll_ops.Rollout(module_env, Agent(), n_step, time_major=time_major)
                ((_, env_outputs), (_, env_outputs_finalizer)) = rollout()

                tf.nest.map_structure(self.assertAllEqual, env_outputs,
                                      expected_env_outputs)
                tf.nest.map_structure(self.assertAllEqual, env_outputs_finalizer,
                                      expected_env_outputs_finalizer)


if __name__ == "__main__":
    tf.test.main()
