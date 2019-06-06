from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from pyoneer import metrics


class MetricsTest(tf.test.TestCase):
    def test_mape_fn(self):
        y_true = tf.constant([[0.2, 0.1], [0.3, 0.2], [0.1, 0.2]], dtype=tf.float32)
        y_pred = tf.constant([[0.1, 0.1], [0.2, 0.1], [0.2, 0.2]], dtype=tf.float32)

        result = metrics.mape(y_true, y_pred)
        expected = tf.constant([0.25, 0.416667, 0.5], dtype=tf.float32)
        self.assertAllClose(result, expected)

    def test_smape_fn(self):
        y_true = tf.constant([[0.2, 0.1], [0.3, 0.2], [0.1, 0.2]], dtype=tf.float32)
        y_pred = tf.constant([[0.1, 0.1], [0.2, 0.1], [0.2, 0.2]], dtype=tf.float32)

        result = metrics.smape(y_true, y_pred)
        expected = tf.constant([0.333333, 0.533333, 0.333333], dtype=tf.float32)
        self.assertAllClose(result, expected)

    def test_mape(self):
        y_true = tf.constant([[0.2, 0.1], [0.3, 0.2], [0.1, 0.2]], dtype=tf.float32)
        y_pred = tf.constant([[0.1, 0.1], [0.2, 0.1], [0.2, 0.2]], dtype=tf.float32)
        sample_weight = tf.constant([1.0, 0.0, 1.0], dtype=tf.float32)

        metric = metrics.MAPE()
        metric.update_state(y_true, y_pred, sample_weight=sample_weight)
        self.assertAllClose(metric.result(), 0.375)

    def test_smape(self):
        y_true = tf.constant([[0.3, 0.1], [0.3, 0.3], [0.1, 0.2]], dtype=tf.float32)
        y_pred = tf.constant([[0.1, 0.1], [0.2, 0.1], [0.3, 0.2]], dtype=tf.float32)
        sample_weight = tf.constant([1.0, 0.0, 1.0], dtype=tf.float32)

        metric = metrics.SMAPE()
        metric.update_state(y_true, y_pred, sample_weight=sample_weight)
        self.assertAllClose(metric.result(), 0.5)


if __name__ == "__main__":
    tf.test.main()
