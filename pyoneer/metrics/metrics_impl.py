from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def mape(labels, predictions, weights=None):
    """
    Calculates the mean absolute percentage error of the predicted values to
    the true values. `labels`, `predictions` should be of the same type and
    shape, while `weights` (if given; defaults to `None`) should be either the
    same shape or a 1-D array of the same length (shape dimension 0) as
    `labels` and `predictions`.

    Args:
        labels: Tensor of ground-truth values.
        predictions: Tensor of predicted values.
        weights: Optional weights tensor.
    """
    assert labels.shape[0] == predictions.shape[0]

    if weights is not None:
        assert weights.shape[0] == labels.shape[0]

    errors = tf.abs((predictions - labels) / labels)

    if weights is not None:
        weights /= tf.reduce_sum(weights, axis=0, keepdims=True)
        errors = tf.reduce_sum(errors * weights, axis=0)
    else:
        errors = tf.reduce_mean(errors, axis=0)

    return errors


def smape(labels, predictions, weights=None):
    """
    Calculates the symmetric mean absolute percentage error of the predicted
    values to the true values. `labels`, `predictions` should be of the same
    type and shape, while `weights` (if given; defaults to `None`) should be
    either the same shape or a 1-D array of the same length (shape dimension 0)
    as `labels` and `predictions`.

    The SMAPE is calculated as:

        2 * mean(|predictions - labels| / (|predictions| + |labels|))

    Therefore, it is bounded in the range of [0, 2].

    Args:
        labels: Tensor of ground-truth values.
        predictions: Tensor of predicted values.
        weights: Optional weights tensor.
    """
    assert labels.shape[0] == predictions.shape[0]

    if weights is not None:
        assert weights.shape[0] == labels.shape[0]

    errors = 2 * tf.abs(predictions - labels) / (tf.abs(predictions) + tf.abs(labels))

    if weights is not None:
        weights /= tf.reduce_sum(weights, axis=0, keepdims=True)
        errors = tf.reduce_sum(errors * weights, axis=0)
    else:
        errors = tf.reduce_mean(errors, axis=0)

    return errors


class MAPE(tf.metrics.Mean):
    """Calculates the mean absolute percentage error of predicted values to the
    actual ground-truth values.

    Attributes:
        name: name of the MAPE object.
        dtype: data type of the tensor.
    """

    def __init__(self, name=None, dtype=tf.float32):
        """Inits MAPE class with name and dtype."""
        super(MAPE, self).__init__(name=name, dtype=dtype)

    def call(self, labels, predictions, weights=None):
        """Accumulates MAPE statistics. `labels` and `predictions` should have
        the same shape and type.

        Args:
            labels: Tensor with the true labels for each example.  One example
                per element of the Tensor.
            predictions: Tensor with the predicted label for each example.
            weights: Optional weighting of each example. Defaults to 1.

        Returns:
            The arguments, for easy chaining.
        """
        tf.assert_equal(
            labels.shape,
            predictions.shape,
            message="shapes of labels and predictions must be equal",
        )
        errors = tf.abs((predictions - labels) / labels)
        super(MAPE, self).call(errors, weights=weights)
        if weights is None:
            return labels, predictions
        else:
            return labels, predictions, weights


class SMAPE(tf.metrics.Mean):
    """
    Calculates the symmetric mean absolute percentage error of predicted
    values to the actual ground-truth values. The SMAPE is calculated as:

        2 * mean(|predictions - labels| / (|predictions| + |labels|))

    Therefore, it is bounded in the range of [0, 2].

    Attributes:
        name: name of the SMAPE object.
        dtype: data type of the tensor.
    """

    def __init__(self, name=None, dtype=tf.float32):
        super(SMAPE, self).__init__(name=name, dtype=dtype)

    def call(self, labels, predictions, weights=None):
        """
        Accumulates SMAPE statistics. `labels` and `predictions` should have
        the same shape and type.

        Args:
            labels: Tensor with the true labels for each example.  One example
                per element of the Tensor.
            predictions: Tensor with the predicted label for each example.
            weights: Optional weighting of each example. Defaults to 1.

        Returns:
            The arguments, for easy chaining.
        """
        tf.assert_equal(
            labels.shape,
            predictions.shape,
            message="shapes of labels and predictions must be equal",
        )
        errors = (
            2 * tf.abs(predictions - labels) / (tf.abs(predictions) + tf.abs(labels))
        )
        super(SMAPE, self).call(errors, weights=weights)
        if weights is None:
            return labels, predictions
        else:
            return labels, predictions, weights
