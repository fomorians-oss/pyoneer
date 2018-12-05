from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.platform import test
from tensorflow.python.framework import test_util

import trfl

from pyoneer.rl.agents import double_q_lambda_agent_impl


class _TestValue(tf.keras.Model):

    def __init__(self, output_size):
        super(_TestValue, self).__init__()
        self.linear = tf.layers.Dense(output_size)

    def call(self, inputs, training=False, reset_state=True):
        return self.linear(inputs)


class DoubleQLambdaAgentTest(test.TestCase):
    
    def testComputeLoss(self):
        with context.eager_mode():
            agent = double_q_lambda_agent_impl.DoubleQLambdaAgent(
                value=_TestValue(5),
                target_value=_TestValue(5),
                optimizer=tf.train.GradientDescentOptimizer(1.))

            total_loss = agent.compute_loss(
                states=tf.zeros([4, 2, 5], tf.float32),
                next_states=tf.zeros([4, 2, 5], tf.float32),
                actions=tf.zeros([4, 2], tf.int32),
                rewards=tf.ones([4, 2,], tf.float32),
                weights=tf.ones([4, 2], tf.float32))
            self.assertShapeEqual(tf.zeros([]).numpy(), total_loss)
            self.assertEqual(2, len(agent.loss))

    def testEstimateGradients(self):
        with context.eager_mode():
            agent = double_q_lambda_agent_impl.DoubleQLambdaAgent(
                value=_TestValue(5),
                target_value=_TestValue(5),
                optimizer=tf.train.GradientDescentOptimizer(1.))

            grads_and_vars = agent.estimate_gradients(
                states=tf.zeros([4, 2, 5], tf.float32),
                next_states=tf.zeros([4, 2, 5], tf.float32),
                actions=tf.zeros([4, 2], tf.int32),
                rewards=tf.ones([4, 2], tf.float32),
                weights=tf.ones([4, 2], tf.float32))

            grads, _ = zip(*grads_and_vars)
            variables = agent.trainable_variables
            for grad, var in zip(grads, variables):
                self.assertShapeEqual(tf.shape(grad).numpy(), tf.shape(var))
            self.assertEqual(2, len(agent.loss))

    def testFit(self):
        with context.eager_mode():
            agent = double_q_lambda_agent_impl.DoubleQLambdaAgent(
                value=_TestValue(5),
                target_value=_TestValue(5),
                optimizer=tf.train.GradientDescentOptimizer(1.))

            _ = agent.fit(
                states=tf.zeros([4, 2, 5], tf.float32),
                next_states=tf.zeros([4, 2, 5], tf.float32),
                actions=tf.zeros([4, 2], tf.int32),
                rewards=tf.ones([4, 2], tf.float32),
                weights=tf.ones([4, 2], tf.float32))
            self.assertEqual(2, len(agent.loss))


if __name__ == "__main__":
    test.main()