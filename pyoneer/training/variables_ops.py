# Copyright 2018 The trfl Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def update_target_variables(
    target_variables, source_variables, rate=1.0, use_locking=False
):
    """
    Returns an op to update a list of target variables from source variables.
    The update rule is:

    ```
    target_variable = (1 - rate) * target_variable + rate * source_variable
    ```

    Args:
        target_variables: a list of the variables to be updated.
        source_variables: a list of the variables used for the update.
        rate: weight used to gate the update. The permitted range is
            0 < rate <= 1, with small rate representing an incremental update,
            and rate == 1 representing a full update (that is, a straight
            copy).
        use_locking: use `tf.Variable.assign`'s locking option when assigning
            source variable values to target variables.
    Raises:
        TypeError: when rate is not a Python float
        ValueError: when rate is out of range, or the source and target
            variables have different numbers or shapes.
    Returns:
        An op that executes all the variable updates.
    """
    if not isinstance(rate, float):
        raise TypeError("Tau has wrong type (should be float) {}".format(rate))

    if not 0.0 < rate <= 1.0:
        raise ValueError("Invalid parameter rate {}".format(rate))

    if len(target_variables) != len(source_variables):
        raise ValueError(
            "Number of target variables {} is not the same as "
            "number of source variables {}".format(
                len(target_variables), len(source_variables)
            )
        )

    same_shape = all(
        target_variable.shape == source_variable.shape
        for target_variable, source_variable in zip(target_variables, source_variables)
    )
    if not same_shape:
        raise ValueError(
            "Target variables do not have the same shape as source variables."
        )

    def update_op(target_variable, source_variable, rate):
        if rate < 1.0:
            return target_variable.assign(
                rate * source_variable + (1 - rate) * target_variable,
                use_locking=use_locking,
            )
        else:
            return target_variable.assign(source_variable, use_locking=use_locking)

    update_ops = [
        update_op(target_var, source_var, rate)
        for target_var, source_var in zip(target_variables, source_variables)
    ]
    return tf.group(*update_ops)
