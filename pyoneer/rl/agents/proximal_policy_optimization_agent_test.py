from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.platform import test
from tensorflow.python.framework import test_util

import tensorflow_probability as tfp

import trfl

from pyoneer.rl.agents import proximal_policy_optimization_agent_impl


class _TestPolicy(tf.keras.Model):

    def __init__(self, action_size):
        super(_TestPolicy, self).__init__()
        self.linear = tf.layers.Dense(action_size)

    def call(self, inputs, training=False, reset_state=True):
        return tfp.distributions.MultivariateNormalDiag(self.linear(inputs))


class _TestValue(tf.keras.Model):

    def __init__(self):
        super(_TestValue, self).__init__()
        self.linear = tf.layers.Dense(1)

    def call(self, inputs, training=False, reset_state=True):
        return self.linear(inputs)


class ProximalPolicyOptimizationAgentTest(test.TestCase):

    def testComputeLoss(self):
        with context.eager_mode():
            agent = proximal_policy_optimization_agent_impl.ProximalPolicyOptimizationAgent(
                policy=_TestPolicy(5),
                behavioral_policy=_TestPolicy(5),
                value=_TestValue(),
                optimizer=tf.train.GradientDescentOptimizer(1.))

            total_loss = agent.compute_loss(
                states=tf.zeros([4, 2, 5], tf.float32),
                actions=tf.zeros([4, 2, 5], tf.float32),
                rewards=tf.ones([4, 2,], tf.float32),
                weights=tf.ones([4, 2], tf.float32))
            self.assertShapeEqual(tf.zeros([]).numpy(), total_loss)
            self.assertEqual(4, len(agent.loss))

    def testEstimateGradients(self):
        with context.eager_mode():
            agent = proximal_policy_optimization_agent_impl.ProximalPolicyOptimizationAgent(
                policy=_TestPolicy(5),
                behavioral_policy=_TestPolicy(5),
                value=_TestValue(),
                optimizer=tf.train.GradientDescentOptimizer(1.))

            grads_and_vars = agent.estimate_gradients(
                states=tf.zeros([4, 2, 5], tf.float32),
                actions=tf.zeros([4, 2, 5], tf.float32),
                rewards=tf.ones([4, 2], tf.float32),
                weights=tf.ones([4, 2], tf.float32))

            grads, _ = zip(*grads_and_vars)
            variables = agent.trainable_variables
            for grad, var in zip(grads, variables):
                self.assertShapeEqual(tf.shape(grad).numpy(), tf.shape(var))
            self.assertEqual(4, len(agent.loss))

    def testFit(self):
        with context.eager_mode():
            agent = proximal_policy_optimization_agent_impl.ProximalPolicyOptimizationAgent(
                policy=_TestPolicy(5),
                behavioral_policy=_TestPolicy(5),
                value=_TestValue(),
                optimizer=tf.train.GradientDescentOptimizer(1.))

            _ = agent.fit(
                states=tf.zeros([4, 2, 5], tf.float32),
                actions=tf.zeros([4, 2, 5], tf.float32),
                rewards=tf.ones([4, 2], tf.float32),
                weights=tf.ones([4, 2], tf.float32))
            self.assertEqual(4, len(agent.loss))


if __name__ == "__main__":
    test.main()