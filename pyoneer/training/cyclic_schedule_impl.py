from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class CyclicSchedule:
    """
    Applies a cyclic learning rate scheduling. One cycle during such this
    schedule consists of linearly increasing the learning rate for `step_size`
    steps from `minval` to `maxval`, then linearly decreasing the learning rate
    for another `step_size` steps from `maxval` to (approximately) 0. Note that
    the scheduler is unaware of the total number of planned training steps, so
    be sure to pick `step_size` and set `global_step` accordingly.


    Args:
        minval: starting learning rate.
        maxval: maximum learning rate.
        step_size: number of optimization steps in half a cycle.
        global_step: Tensorflow variable keeping track of the current
            optimization step.
    """

    def __init__(self, minval, maxval, step_size, global_step=None):
        self.minval = minval
        self.maxval = maxval
        self.step_size = step_size
        if global_step is None:
            self.global_step = tf.train.get_or_create_global_step()
        else:
            self.global_step = global_step
        self.value = tf.Variable(minval, trainable=False)

    def __call__(self):
        step = tf.to_float(self.global_step - 1)
        cycle = 1 + step // (2 * self.step_size)
        x = step / self.step_size - 2 * cycle + 1
        x_abs = tf.abs(x)
        # decay the downward half towards 0 instead of minval
        minval = tf.where(x < 0, self.minval, 0.0)
        vrange = self.maxval - minval
        value = minval + vrange * tf.maximum(0.0, 1 - x_abs)
        self.value.assign(value)
        return value
