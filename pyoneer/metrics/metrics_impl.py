from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.keras.metrics import MeanMetricWrapper


def mape(labels, predictions):
    """
    Calculates the mean absolute percentage error of the predicted values to
    the true values. `labels`, `predictions` should be of the same type and
    shape.

    Args:
        labels: Tensor of ground-truth values.
        predictions: Tensor of predicted values.
    """
    errors = tf.abs((predictions - labels) / labels)
    return errors


def smape(labels, predictions):
    """
    Calculates the symmetric mean absolute percentage error of the predicted
    values to the true values. `labels`, `predictions` should be of the same
    type and shape.

    The SMAPE is calculated as:

        2 * mean(|predictions - labels| / (|predictions| + |labels|))

    Therefore, it is bounded in the range of [0, 2].

    Args:
        labels: Tensor of ground-truth values.
        predictions: Tensor of predicted values.
    """
    errors = 2 * tf.abs(predictions - labels) / (tf.abs(predictions) + tf.abs(labels))
    return errors


class MAPE(MeanMetricWrapper):
    """
    Calculates the mean absolute percentage error of predicted values to the
    actual ground-truth values.

    Attributes:
        name: name of the MAPE object.
        dtype: data type of the tensor.
    """

    def __init__(self, name="MAPE", dtype=None):
        """Inits MAPE class with name and dtype."""
        super(MAPE, self).__init__(mape, name, dtype=dtype)


class SMAPE(MeanMetricWrapper):
    """
    Calculates the symmetric mean absolute percentage error of predicted
    values to the actual ground-truth values. The SMAPE is calculated as:

        2 * mean(|predictions - labels| / (|predictions| + |labels|))

    Therefore, it is bounded in the range of [0, 2].

    Attributes:
        name: name of the SMAPE object.
        dtype: data type of the tensor.
    """

    def __init__(self, name="SMAPE", dtype=None):
        super(SMAPE, self).__init__(smape, name, dtype=dtype)
