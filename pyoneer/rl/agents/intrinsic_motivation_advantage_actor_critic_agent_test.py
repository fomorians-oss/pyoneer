from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.platform import test
from tensorflow.python.framework import test_util

import tensorflow_probability as tfp

from pyoneer.rl.agents import intrinsic_motivation_advantage_actor_critic_agent_impl


class _TestPolicy(tf.keras.Model):

    def __init__(self, action_size):
        super(_TestPolicy, self).__init__()
        self.linear = tf.layers.Dense(action_size)

    def call(self, inputs, training=False, reset_state=True):
        return tfp.distributions.MultivariateNormalDiag(self.linear(inputs))


class _TestValue(tf.keras.Model):

    def __init__(self, num_units):
        super(_TestValue, self).__init__()
        self.linear = tf.layers.Dense(num_units)

    def call(self, inputs, training=False, reset_state=True):
        return self.linear(inputs)


class _TestModel(tf.keras.Model):

    def __init__(self, num_units):
        super(_TestModel, self).__init__()
        self.linear = tf.layers.Dense(num_units)

    def call(self, states, actions, training=False, reset_state=True):
        return self.linear(tf.concat([states, actions], axis=-1))


class IntrinsicMotivationAdvantageActorCriticAgentTest(test.TestCase):

    def testComputeLoss(self):
        with context.eager_mode():
            agent = intrinsic_motivation_advantage_actor_critic_agent_impl.IntrinsicMotivationAdvantageActorCriticAgent(
                policy=_TestPolicy(5),
                value=_TestValue(1),
                forward_model=_TestModel(5),
                optimizer=tf.train.GradientDescentOptimizer(1.),
                forward_optimizer=tf.train.GradientDescentOptimizer(1.))

            total_loss = agent.compute_loss(
                states=tf.zeros([4, 2, 5], tf.float32),
                next_states=tf.zeros([4, 2, 5], tf.float32),
                actions=tf.zeros([4, 2, 5], tf.float32),
                rewards=tf.ones([4, 2], tf.float32),
                weights=tf.ones([4, 2], tf.float32))
            self.assertShapeEqual(tf.zeros([]).numpy(), total_loss)
            self.assertEqual(5, len(agent.loss))

    def testEstimateGradients(self):
        with context.eager_mode():
            agent = intrinsic_motivation_advantage_actor_critic_agent_impl.IntrinsicMotivationAdvantageActorCriticAgent(
                policy=_TestPolicy(5),
                value=_TestValue(1),
                forward_model=_TestModel(5),
                optimizer=tf.train.GradientDescentOptimizer(1.),
                forward_optimizer=tf.train.GradientDescentOptimizer(1.))

            policy_value_grads_and_vars, forward_grads_and_vars = agent.estimate_gradients(
                states=tf.zeros([4, 2, 5], tf.float32),
                next_states=tf.zeros([4, 2, 5], tf.float32),
                actions=tf.zeros([4, 2, 5], tf.float32),
                rewards=tf.ones([4, 2], tf.float32),
                weights=tf.ones([4, 2], tf.float32))

            grads, _ = zip(*policy_value_grads_and_vars)
            variables = agent.policy.trainable_variables + agent.value.trainable_variables
            for grad, var in zip(grads, variables):
                self.assertShapeEqual(tf.shape(grad).numpy(), tf.shape(var))
            
            grads, _ = zip(*forward_grads_and_vars)
            variables = agent.forward_model.trainable_variables
            for grad, var in zip(grads, variables):
                self.assertShapeEqual(tf.shape(grad).numpy(), tf.shape(var))
            self.assertEqual(5, len(agent.loss))

    def testFit(self):
        with context.eager_mode():
            agent = intrinsic_motivation_advantage_actor_critic_agent_impl.IntrinsicMotivationAdvantageActorCriticAgent(
                policy=_TestPolicy(5),
                value=_TestValue(1),
                forward_model=_TestModel(5),
                optimizer=tf.train.GradientDescentOptimizer(1.),
                forward_optimizer=tf.train.GradientDescentOptimizer(1.))

            _ = agent.fit(
                states=tf.zeros([4, 2, 5], tf.float32),
                next_states=tf.zeros([4, 2, 5], tf.float32),
                actions=tf.zeros([4, 2, 5], tf.float32),
                rewards=tf.ones([4, 2], tf.float32),
                weights=tf.ones([4, 2], tf.float32))
            self.assertEqual(5, len(agent.loss))


if __name__ == "__main__":
    test.main()