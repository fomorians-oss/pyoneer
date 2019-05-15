from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.eager import context
from tensorflow.python.platform import test

from pyoneer.metrics import MAPE, SMAPE


class MetricsTest(test.TestCase):
    def test_mape(self):
        with context.eager_mode():
            labels = tf.constant([[0.2, 0.1], [0.3, 0.2], [0.1, 0.2]])
            predictions = tf.constant([[0.1, 0.1], [0.2, 0.1], [0.2, 0.2]])
            weights = tf.constant([[1.0, 1.0], [0.0, 0.0], [1.0, 1.0]])

            mape = MAPE()
            mape(predictions, labels, sample_weight=weights)
            self.assertAllClose(mape.result(), 0.375)

    def test_smape(self):
        with context.eager_mode():
            labels = tf.constant([[0.3, 0.1], [0.3, 0.3], [0.1, 0.2]])
            predictions = tf.constant([[0.1, 0.1], [0.2, 0.1], [0.3, 0.2]])
            weights = tf.constant([[1.0, 1.0], [0.0, 0.0], [1.0, 1.0]])

            smape = SMAPE()
            smape(predictions, labels, sample_weight=weights)
            self.assertAllClose(smape.result(), 0.5)


if __name__ == "__main__":
    test.main()
