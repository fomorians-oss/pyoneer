from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class CyclicSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Cyclic learning rate schedule. One cycle during such this
    schedule consists of linearly increasing the learning rate for `step_size`
    steps from `minval` to `maxval`, then linearly decreasing the learning rate
    for another `step_size` steps from `maxval` to (approximately) 0. Note that
    the scheduler is unaware of the total number of planned training steps, so
    be sure to pick `step_size` accordingly.

    Args:
        minval: starting learning rate.
        maxval: maximum learning rate.
        step_size: number of optimization steps in half a cycle.
    """

    def __init__(self, minval, maxval, step_size, decay_to_zero=False):
        self.minval = minval
        self.maxval = maxval
        self.step_size = step_size
        self.decay_to_zero = decay_to_zero

    def get_config(self):
        return {
            "minval": self.minval,
            "maxval": self.maxval,
            "step_size": self.step_size,
        }

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        cycle = 1 + step // (2 * self.step_size)
        x = step / self.step_size - 2 * cycle + 1
        x_abs = tf.abs(x)
        # decay the downward half towards 0 instead of minval
        if self.decay_to_zero:
            minval = tf.where(x < 0, self.minval, 0.0)
        else:
            minval = self.minval
        vrange = self.maxval - minval
        return minval + vrange * tf.maximum(0.0, 1 - x_abs)
