from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.python.eager import context
from tensorflow.python.platform import test

import pyoneer.metrics as metrics


class MetricsTest(test.TestCase):
    def test_mape_fn(self):
        with context.eager_mode():
            labels = tf.constant([[0.2, 0.1], [0.3, 0.2], [0.1, 0.2]])
            predictions = tf.constant([[0.1, 0.1], [0.2, 0.1], [0.2, 0.2]])
            weights = tf.constant([[1.0], [0.0], [1.0]])
            self.assertAllClose(
                metrics.mape(predictions, labels, weights=weights),
                tf.constant([0.75, 0.0]))

    def test_smape_fn(self):
        with context.eager_mode():
            labels = tf.constant([[0.3, 0.1], [0.3, 0.3], [0.1, 0.2]])
            predictions = tf.constant([[0.1, 0.1], [0.2, 0.1], [0.3, 0.2]])
            weights = tf.constant([[1.0], [0.0], [1.0]])
            self.assertAllClose(
                metrics.smape(predictions, labels, weights=weights),
                tf.constant([1.0, 0.0]))

    def test_mape_class(self):
        with context.eager_mode():
            labels = tf.constant([[0.2, 0.1], [0.3, 0.2], [0.1, 0.2]])
            predictions = tf.constant([[0.1, 0.1], [0.2, 0.1], [0.2, 0.2]])
            weights = tf.constant([[1.0, 1.0], [0.0, 0.0], [1.0, 1.0]])

            mape = metrics.MAPE()
            mape(predictions, labels, weights=weights)
            self.assertAllClose(mape.result(), 0.375)

    def test_smape_class(self):
        with context.eager_mode():
            labels = tf.constant([[0.3, 0.1], [0.3, 0.3], [0.1, 0.2]])
            predictions = tf.constant([[0.1, 0.1], [0.2, 0.1], [0.3, 0.2]])
            weights = tf.constant([[1.0, 1.0], [0.0, 0.0], [1.0, 1.0]])

            smape = metrics.SMAPE()
            smape(predictions, labels, weights=weights)
            self.assertAllClose(smape.result(), 0.5)


if __name__ == '__main__':
    test.main()
