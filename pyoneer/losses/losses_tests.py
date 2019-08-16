from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from pyoneer.losses import losses_impl


class LossesTest(tf.test.TestCase):
    def test_compute_weighted_losses(self):
        losses = tf.constant([[1.0, -1.0, 1.0]])
        sample_weight = tf.constant([[1.0, 0.0, 1.0]])
        actual = losses_impl.compute_weighted_losses(losses, sample_weight)
        expected = 1.0
        self.assertAllEqual(actual, expected)


if __name__ == "__main__":
    tf.test.main()
