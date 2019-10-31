from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf


def update_variables(source_variables, target_variables, rate=1.0, use_locking=False):
    """
    Update a list of target variables from source variables:

        target_variable = (1 - rate) * target_variable + rate * source_variable

    Args:
        target_variables: a list of the variables to be updated.
        source_variables: a list of the variables used for the update.
        rate: weight used to gate the update. The permitted range is `0 < rate <= 1`,
            with small rate representing an incremental update, and `rate == 1`
            representing a full update (that is, a straight copy).
        use_locking: use `tf.Variable.assign`'s locking option when assigning
            source variable values to target variables.
    Raises:
        TypeError: when rate is not a `float`
        ValueError: when rate is out of range, or the source and target variables
            have different numbers or shapes.
        ValueError: when the length of `source_variables` does not equal the length
            of `target_variables`.
        ValueError: when the shapes of `source_variables` does not equal the shapes
            of `target_variables`.
    """
    if not isinstance(rate, float):
        raise TypeError("Rate should be a float but got: {}".format(type(rate)))
    if not (0.0 < rate <= 1.0):
        raise ValueError(
            "Rate should be greater than 0 and less than or "
            "equal to 1 but got: {}".format(rate)
        )
    if len(source_variables) != len(target_variables):
        raise ValueError(
            "Number of source variables {} is not the same as "
            "number of target variables {}".format(
                len(source_variables), len(target_variables)
            )
        )

    same_shape = all(
        source_var.get_shape() == target_var.get_shape()
        for source_var, target_var in zip(source_variables, target_variables)
    )
    if not same_shape:
        raise ValueError(
            "Target variables don't have the same shape as source variables"
        )

    for source_var, target_var in zip(source_variables, target_variables):
        if rate == 1.0:
            target_var.assign(source_var, use_locking)
        else:
            target_var.assign(
                rate * source_var + (1.0 - rate) * target_var, use_locking
            )
