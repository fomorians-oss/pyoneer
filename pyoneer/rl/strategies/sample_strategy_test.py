from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.python.eager import context
from tensorflow.python.platform import test

from pyoneer.rl.strategies import sample_strategy_impl


class SampleStrategyTest(test.TestCase):
    def testSampleStrategy(self):
        with context.eager_mode():
            logits = [[0.0, 1.0, -2.0], [0.0, 1.0, 2.0]]

            def policy(x):
                return tfp.distributions.Categorical(logits=x)

            expected_samples = tf.constant([1, 2])
            expected_policy_samples = policy(logits).mode()
            self.assertAllEqual(expected_policy_samples, expected_samples)

            sampler = sample_strategy_impl.SampleStrategy(policy)
            samples = sampler(logits)
            self.assertShapeEqual(samples.numpy(), expected_policy_samples)


if __name__ == "__main__":
    test.main()
