from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.contrib.eager as tfe

from tensorflow.python.eager import context
from tensorflow.python.platform import test

from pyoneer.rl.strategies import ornstein_uhlenbeck_strategy_impl


class OrnsteinUhlenbeckStrategyTest(test.TestCase):

    def testOrnsteinUhlenbeckStrategy(self):
        with context.eager_mode():
            logits = [[0., 0.], 
                      [0., 0.]]
            policy = lambda x: tfp.distributions.Normal(
                loc=x, 
                scale=0.)
            expected_samples = tf.constant([[0., 0.], [0., 0.]])
            expected_policy_samples = policy(logits).mode()
            self.assertAllEqual(expected_policy_samples, expected_samples)

            ornstein_uhlenbeck = ornstein_uhlenbeck_strategy_impl.OrnsteinUhlenbeckStrategy(policy)
            for _ in range(5):
                samples = ornstein_uhlenbeck(logits)
                self.assertShapeEqual(samples.numpy(), expected_policy_samples)
            ornstein_uhlenbeck.reset_state()
            for _ in range(5):
                samples = ornstein_uhlenbeck(logits)
                self.assertShapeEqual(samples.numpy(), expected_policy_samples)


if __name__ == '__main__':
    test.main()
