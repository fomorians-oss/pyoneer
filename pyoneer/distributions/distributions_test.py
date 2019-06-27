from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_probability as tfp

from pyoneer.distributions.distributions_impl import MultiCategorical


class DistributionsTest(tf.test.TestCase):
    def test_multicategorical_log_prob(self):
        dist1 = tfp.distributions.Categorical(logits=[[0.1, 0.2, 0.3]])
        dist2 = tfp.distributions.Categorical(logits=[[0.3, 0.2, 0.1]])

        distribution = MultiCategorical([dist1, dist2])
        output = distribution.log_prob([[2, 0]])

        expected = tf.constant([-2.0038857], dtype=tf.float32)
        self.assertAllClose(output, expected)

    def test_multicategorical_entropy(self):
        dist1 = tfp.distributions.Categorical(logits=[[0.1, 0.2, 0.3]])
        dist2 = tfp.distributions.Categorical(logits=[[0.3, 0.2, 0.1]])

        distribution = MultiCategorical([dist1, dist2])
        output = distribution.entropy()

        expected = tf.constant([2.1905746], dtype=tf.float32)
        self.assertAllClose(output, expected)

    def test_multicategorical_mode(self):
        dist1 = tfp.distributions.Categorical(logits=[[0.1, 0.2, 0.3]])
        dist2 = tfp.distributions.Categorical(logits=[[0.3, 0.2, 0.1]])

        distribution = MultiCategorical([dist1, dist2])
        output = distribution.mode()

        expected = tf.constant([[2, 0]], dtype=tf.int32)
        self.assertAllClose(output, expected)


if __name__ == "__main__":
    tf.test.main()
