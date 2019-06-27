from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_probability as tfp

from pyoneer.rl.strategies import strategies_impl


class StrategiesTest(tf.test.TestCase):
    def test_epsilon_greedy(self):
        logits = [[0.0, 1.0, -2.0], [0.0, 1.0, 2.0]]

        def policy(x):
            return tfp.distributions.Categorical(logits=x)

        expected_samples = tf.constant([1, 2])
        expected_policy_samples = policy(logits).mode()
        self.assertAllEqual(expected_policy_samples, expected_samples)

        epsilon_greedy = strategies_impl.EpsilonGreedy(policy, 0.0)
        samples = epsilon_greedy(logits)
        self.assertAllEqual(samples, expected_policy_samples)

    def test_sample(self):
        logits = [[0.0, 1.0, -2.0], [0.0, 1.0, 2.0]]

        def policy(x):
            return tfp.distributions.Categorical(logits=x)

        expected_samples = tf.constant([1, 2])
        expected_policy_samples = policy(logits).mode()
        self.assertAllEqual(expected_policy_samples, expected_samples)

        sampler = strategies_impl.Sample(policy)
        samples = sampler(logits)
        self.assertShapeEqual(samples.numpy(), expected_policy_samples)

    def test_mode(self):
        logits = [[0.0, 1.0, -2.0], [0.0, 1.0, 2.0]]

        def policy(x):
            return tfp.distributions.Categorical(logits=x)

        expected_samples = tf.constant([1, 2])
        expected_policy_samples = policy(logits).mode()
        self.assertAllEqual(expected_policy_samples, expected_samples)

        sampler = strategies_impl.Mode(policy)
        samples = sampler(logits)
        self.assertShapeEqual(samples.numpy(), expected_policy_samples)


if __name__ == "__main__":
    tf.test.main()
