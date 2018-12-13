from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.contrib.eager.python import metrics
from tensorflow.python.ops import array_ops, check_ops

def mean_absolute_percentage_error(y_true, y_pred,
                                sample_weight=None,
                                multioutput='raw_values',
                                mask_invalid=False):
    """Calculates the mean absolute percentage error of the predicted values to
    the true values. `y_true`, `y_pred` should be of the same type and shape,
    while `sample_weight` (if given; defaults to `None`) should be either the
    same shape or a 1-D array of the same length (shape dimension 0) as `y_true`
    and `y_pred`.

    Args:
        y_true: array of the true values.
        y_pred: array of the predicted values.
        multioutput: a string. If `'raw_values'`, returns a full set of errors
            in case of multioutput input (equal to shape dimension -1). If
            `'uniform_average'`, returns a uniformly weighted average error.
        mask_invalid: if `True`, masks any invalid values in `y_pred` and
            `y_true`, otherwise errors will be thrown if any are present.
            Defaults to `False`.
    """
    if sample_weight is not None:
        assert len(sample_weight) == len(y_true) == len(y_pred)
    if mask_invalid:
        # Mask invalid values to avoid throwing an error:
        apes = np.ma.masked_invalid(np.abs((y_pred - y_true) / y_true))
        output_errors = np.ma.average(apes, axis=0, weights=sample_weight)
    else:
        # Don't mask invalid values so errors are thrown if they're present
        apes = np.abs((y_pred - y_true) / y_true)
        output_errors = np.average(apes, axis=0, weights=sample_weight)
    if multioutput == 'raw_values':
        return output_errors
    elif multioutput == 'uniform_average':
        return np.average(output_errors, weights=None)

def symmetric_mean_absolute_percentage_error(y_true, y_pred,
                                            sample_weight=None,
                                            multioutput='raw_values',
                                            mask_invalid=False):
    """Calculates the symmetric mean absolute percentage error of the predicted
    values to the true values. `y_true`, `y_pred` should be of the same type
    and shape, while `sample_weight` (if given; defaults to `None`) should be
    either the same shape or a 1-D array of the same length (shape dimension 0)
    as `y_true` and `y_pred`.

    The SMAPE is calculated as:

        2 * mean(|predictions - labels| / (|predictions| + |labels|))
        
    Therefore, it is bounded in the range of [0, 2].

    Args:
        y_true: array of the true values.
        y_pred: array of the predicted values.
        multioutput: a string. If `'raw_values'`, returns a full set of errors
            in case of multioutput input (equal to shape dimension -1). If
            `'uniform_average'`, returns a uniformly weighted average error.
        mask_invalid: if `True`, masks any invalid values in `y_pred` and
            `y_true`, otherwise errors will be thrown if any are present.
            Defaults to `False`.
    """
    if sample_weight is not None:
        assert len(sample_weight) == len(y_true) == len(y_pred)
    sapes = 2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))
    if mask_invalid:
        # Mask invalid values to avoid throwing an error:
        sapes = np.ma.masked_invalid(sapes)
        output_errors = np.ma.average(sapes, axis=0, weights=sample_weight)
    else:
        # Don't mask invalid values so errors are thrown if they're present:
        output_errors = np.average(sapes, axis=0, weights=sample_weight)
    if multioutput == 'raw_values':
        return output_errors
    elif multioutput == 'uniform_average':
        return np.average(output_errors, weights = None)

class MeanAbsolutePercentageError(metrics.Mean):
    """Calculates the mean absolute percentage error of predicted values to the
    actual ground-truth values.

    Attributes:
        name: name of the MeanAbsolutePercentageError object.
        dtype: data type of the tensor.
    """
    def __init__(self, name=None, dtype=tf.float32):
        """Inits MeanAbsolutePercentageError class with name and dtype."""
        super(MeanAbsolutePercentageError, self).__init__(name=name, dtype=dtype)

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
        check_ops.assert_equal(
            array_ops.shape(labels), array_ops.shape(predictions),
            message="Shapes of labels and predictions are unequal")
        ape = tf.abs((predictions - labels) / labels)
        super(MeanAbsolutePercentageError, self).call(ape, weights=weights)
        if weights is None:
            return labels, predictions
        else:
            return labels, predictions, weights


class SymmetricMeanAbsolutePercentageError(metrics.Mean):
    """Calculates the symmetric mean absolute percentage error of predicted
    values to the actual ground-truth values. The SMAPE is calculated as:

        2 * mean(|predictions - labels| / (|predictions| + |labels|))
        
    Therefore, it is bounded in the range of [0, 2].

    Attributes:
        name: name of the SymmetricMeanAbsolutePercentageError object.
        dtype: data type of the tensor.
    """
    def __init__(self, name=None, dtype=tf.float32):
        """Inits SymmetricMeanAbsolutePercentageError class with name and dtype."""
        super(SymmetricMeanAbsolutePercentageError, self).__init__(name=name, dtype=dtype)

    def call(self, labels, predictions, weights=None):
        """Accumulates SMAPE statistics. `labels` and `predictions` should have
        the same shape and type.

        Args:
            labels: Tensor with the true labels for each example.  One example
                per element of the Tensor.
            predictions: Tensor with the predicted label for each example.
            weights: Optional weighting of each example. Defaults to 1.
        
        Returns:
            The arguments, for easy chaining.
        """
        check_ops.assert_equal(
            array_ops.shape(labels), array_ops.shape(predictions),
            message="Shapes of labels and predictions are unequal")
        sapes = 2 * tf.abs(predictions - labels) / (tf.abs(predictions) + tf.abs(labels))
        super(SymmetricMeanAbsolutePercentageError, self).call(sapes, weights=weights)
        if weights is None:
            return labels, predictions
        else:
            return labels, predictions, weights
