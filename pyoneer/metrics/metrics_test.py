from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

from tensorflow.python.eager import context
from tensorflow.python.platform import test

import pyoneer.metrics as metrics

class MetricsTest(test.TestCase):
    def test_mape_fn(self):
        y_true = np.asarray([0.1, 0.2])
        y_pred = np.asarray([0.1, 0.2])
        self.assertAllEqual(metrics.mape(y_pred, y_true), 0)

        y_true = np.asarray([[0.2, 0.1], [0.3, 0.2], [0.1, 0.2]])
        y_pred = np.asarray([[0.1, 0.1], [0.2, 0.1], [0.2, 0.2]])
        weights = np.asarray([1, 0, 1])
        self.assertAllEqual(
            metrics.mape(y_pred, y_true), np.asarray([2/3, 1/3]))
        self.assertAllEqual(
            metrics.mape(y_pred, y_true, multioutput='uniform_average'), 0.5)
        self.assertAllEqual(
            metrics.mape(y_pred, y_true, sample_weight=weights),
            np.asarray([0.75, 0]))
        self.assertAllEqual(
            metrics.mape(
                y_pred, y_true, 
                multioutput='uniform_average',
                sample_weight=weights),
            0.375)

        y_true = np.asarray([[0.2, 0.1], [0.3, np.nan], [0.1, 0.2]])
        y_pred = np.asarray([[0.1, 0.1], [np.nan, 0.1], [0.2, 0.2]])
        self.assertAllEqual(
            metrics.mape(y_pred, y_true), np.asarray([np.nan, np.nan]))
        self.assertAllEqual(
            metrics.mape(y_pred, y_true, mask_invalid=True),
            np.asarray([0.75, 0]))
        self.assertAllEqual(
            metrics.mape(
                y_pred, y_true,
                multioutput='uniform_average',
                mask_invalid=True),
            0.375)

    def test_smape_fn(self):
        y_true = np.asarray([0.1, 0.2])
        y_pred = np.asarray([0.1, 0.2])
        self.assertAllEqual(metrics.smape(y_pred, y_true), 0)

        y_true = np.asarray([[0.3, 0.1], [0.3, 0.3], [0.1, 0.2]])
        y_pred = np.asarray([[0.1, 0.1], [0.2, 0.1], [0.3, 0.2]])
        weights = np.asarray([1, 0, 1])
        self.assertAllClose(
            metrics.smape(y_pred, y_true), np.asarray([0.8, 1/3]))
        self.assertAllClose(
            metrics.smape(y_pred, y_true, multioutput='uniform_average'),
            np.average([0.8, 1/3]))
        self.assertAllClose(
            metrics.smape(y_pred, y_true, sample_weight=weights),
            np.asarray([1., 0.]))
        self.assertAllClose(
            metrics.smape(
                y_pred, y_true, 
                multioutput='uniform_average',
                sample_weight=weights),
            0.5)

        y_true = np.asarray([[0.3, 0.1], [0.3, np.nan], [0.1, 0.2]])
        y_pred = np.asarray([[0.1, 0.1], [np.nan, 0.1], [0.3, 0.2]])
        self.assertAllEqual(
            metrics.smape(y_pred, y_true), np.asarray([np.nan, np.nan]))
        self.assertAllClose(
            metrics.smape(y_pred, y_true, mask_invalid=True),
            np.asarray([1., 0.]))
        self.assertAllClose(
            metrics.smape(
                y_pred, y_true,
                multioutput='uniform_average',
                mask_invalid=True),
            0.5)

    def test_mape_class(self):
        with context.eager_mode():
            y_true = tf.constant([[0.2, 0.1], [0.3, 0.2], [0.1, 0.2]])
            y_pred = tf.constant([[0.1, 0.1], [0.2, 0.1], [0.2, 0.2]])
            weights = tf.constant([[1, 1], [0, 0], [1, 1]])

            mape = metrics.MAPE()
            mape(y_pred, y_true)
            self.assertAllEqual(mape.result().numpy(), 0.5)

            mape = metrics.MAPE()
            mape(y_pred, y_true, weights=weights)
            self.assertAllEqual(mape.result().numpy(), 0.375)

    def test_smape_class(self):
        with context.eager_mode():
            y_true = tf.constant([[0.3, 0.1], [0.3, 0.3], [0.1, 0.2]])
            y_pred = tf.constant([[0.1, 0.1], [0.2, 0.1], [0.3, 0.2]])
            weights = tf.constant([[1, 1], [0, 0], [1, 1]])

            smape = metrics.SMAPE()
            smape(y_pred, y_true)
            self.assertAllClose(smape.result().numpy(), np.average([0.8, 1/3]))

            smape = metrics.SMAPE()
            smape(y_pred, y_true, weights=weights)
            self.assertAllClose(smape.result().numpy(), 0.5)


if __name__ == '__main__':
    test.main()
