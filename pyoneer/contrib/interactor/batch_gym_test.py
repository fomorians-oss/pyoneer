from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym

import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.platform import test

from pyoneer.contrib.interactor import batch_gym_impl


class MockSpace(object):
    
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype


class CounterEnv(gym.Env):

    def __init__(self):
        self.observation_space = MockSpace([], tf.float32)
        self.action_space = MockSpace([], tf.float32)
        self._step = 0

    def reset(self):
        self._step = 1
        return self._step

    def step(self, action):
        self._step += 1
        return self._step, 0., False, {}


class BatchGymTest(test.TestCase):

    def testBatchGymMakeSpec(self):
        with context.eager_mode():
            env = batch_gym_impl.batch_gym_make('Pendulum-v0')
            self.assertTrue(isinstance(env, batch_gym_impl.BatchEnv))

    def testBatchGymMakeLambda(self):
        with context.eager_mode():
            env = batch_gym_impl.batch_gym_make(lambda: CounterEnv())
            self.assertTrue(isinstance(env, batch_gym_impl.BatchEnv))

    def testResetLambda(self):
        with context.eager_mode():
            env = batch_gym_impl.batch_gym_make(lambda: CounterEnv())
            state = env.reset(5)
            self.assertAllEqual(state, tf.ones([5], tf.float32))

    def testResetSpec(self):
        with context.eager_mode():
            env = batch_gym_impl.batch_gym_make('Pendulum-v0')
            state = env.reset(5)
            self.assertShapeEqual(
                state.numpy(), tf.ones([5] + list(env.observation_space.shape), tf.float32))

    def testStepLambda(self):
        with context.eager_mode():
            env = batch_gym_impl.batch_gym_make(lambda: CounterEnv())
            _ = env.reset(5)
            next_state, reward, done, _ = env.step([0] * 5)
            self.assertAllEqual(next_state, 1 + tf.ones([5], tf.float32))
            self.assertAllEqual(reward, tf.zeros([5], tf.float32))
            self.assertAllEqual(done, tf.zeros([5], tf.bool))

            next_state, _, _, _ = env.step([0] * 5)
            self.assertAllEqual(next_state, 2 + tf.ones([5], tf.float32))

    def testStepSpec(self):
        with context.eager_mode():
            env = batch_gym_impl.batch_gym_make('Pendulum-v0')
            _ = env.reset(5)
            next_state, reward, done, _ = env.step([[0]] * 5)
            self.assertShapeEqual(
                next_state.numpy(), tf.ones([5] + list(env.observation_space.shape), tf.float32))
            self.assertShapeEqual(reward.numpy(), tf.zeros([5], tf.float32))
            self.assertShapeEqual(done.numpy(), tf.zeros([5], tf.bool))

            next_state, _, _, _ = env.step([[0]] * 5)
            self.assertShapeEqual(
                next_state.numpy(), tf.ones([5] + list(env.observation_space.shape), tf.float32))

    def testStepAndResetLambda(self):
        with context.eager_mode():
            env = batch_gym_impl.batch_gym_make(lambda: CounterEnv())
            _ = env.reset(5)
            _ = env.step([0] * 5)
            state = env.reset(2)
            self.assertAllEqual(state, tf.ones([2], tf.float32))
            state = env.reset(5)
            self.assertAllEqual(state, tf.ones([5], tf.float32))
            state = env.reset(10)
            self.assertAllEqual(state, tf.ones([10], tf.float32))
            state = env.reset(1)
            self.assertAllEqual(state, tf.ones([1], tf.float32))
    
    def testStepAndResetSpec(self):
        with context.eager_mode():
            env = batch_gym_impl.batch_gym_make('Pendulum-v0')
            _ = env.reset(5)
            _ = env.step([[0]] * 5)
            state = env.reset(2)
            self.assertShapeEqual(
                state.numpy(), tf.ones([2] + list(env.observation_space.shape), tf.float32))
            state = env.reset(5)
            self.assertShapeEqual(
                state.numpy(), tf.ones([5] + list(env.observation_space.shape), tf.float32))
            state = env.reset(10)
            self.assertShapeEqual(
                state.numpy(), tf.ones([10] + list(env.observation_space.shape), tf.float32))
            state = env.reset(1)
            self.assertShapeEqual(
                state.numpy(), tf.ones([1] + list(env.observation_space.shape), tf.float32))


if __name__ == "__main__":
    test.main()