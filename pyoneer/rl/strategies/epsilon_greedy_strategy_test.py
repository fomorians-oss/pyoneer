from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.python.eager import context
from tensorflow.python.platform import test

from pyoneer.rl.strategies import epsilon_greedy_strategy_impl


class EpsilonGreedyStrategyTest(test.TestCase):
    def test_epsilon_greedy_strategy(self):
        with context.eager_mode():
            logits = [[0.0, 1.0, -2.0], [0.0, 1.0, 2.0]]

            def policy(x):
                return tfp.distributions.Categorical(logits=x)

            expected_samples = tf.constant([1, 2])
            expected_policy_samples = policy(logits).mode()
            self.assertAllEqual(expected_policy_samples, expected_samples)

            epsilon_greedy = epsilon_greedy_strategy_impl.EpsilonGreedyStrategy(
                policy, 0.0
            )
            samples = epsilon_greedy(logits)
            self.assertAllEqual(samples, expected_policy_samples)


if __name__ == "__main__":
    test.main()
