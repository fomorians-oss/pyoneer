from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.keras.metrics import MeanMetricWrapper


def mape(y_true, y_pred):
    """
    Calculates the mean absolute percentage error of the predicted values to
    the true values. `y_true`, `y_pred` should be of the same type and
    shape.

    Args:
        y_true: Tensor of ground-truth values.
        y_pred: Tensor of predicted values.
    """
    y_pred.get_shape().assert_is_compatible_with(y_true.get_shape())
    if y_true.dtype != y_pred.dtype:
        y_pred = tf.cast(y_pred, y_true.dtype)
    errors = tf.abs((y_pred - y_true) / y_true)
    errors = tf.reduce_mean(errors, axis=-1)
    return errors


def smape(y_true, y_pred):
    """
    Calculates the symmetric mean absolute percentage error of the predicted
    values to the true values. `y_true`, `y_pred` should be of the same
    type and shape.

    The SMAPE is calculated as:

        2 * mean(|y_pred - y_true| / (|y_pred| + |y_true|))

    Therefore, it is bounded in the range of [0, 2].

    Args:
        y_true: Tensor of ground-truth values.
        y_pred: Tensor of predicted values.
    """
    y_pred.get_shape().assert_is_compatible_with(y_true.get_shape())
    if y_true.dtype != y_pred.dtype:
        y_pred = tf.cast(y_pred, y_true.dtype)
    errors = 2 * tf.abs(y_pred - y_true) / (tf.abs(y_pred) + tf.abs(y_true))
    errors = tf.reduce_mean(errors, axis=-1)
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

        2 * mean(|y_pred - y_true| / (|y_pred| + |y_true|))

    Therefore, it's bounded in the range of [0, 2].

    Attributes:
        name: name of the SMAPE object.
        dtype: data type of the tensor.
    """

    def __init__(self, name="SMAPE", dtype=None):
        """Inits SMAPE class with name and dtype."""
        super(SMAPE, self).__init__(smape, name, dtype=dtype)
