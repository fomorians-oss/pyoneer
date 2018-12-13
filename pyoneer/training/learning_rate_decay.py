from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.framework import ops

def cyclic_lr_scheduler(lr_min, lr_max, step_size, global_step, name=None):
    """Applies a cyclic learning rate scheduling. One cycle during such this
    schedule consists of linearly increasing the learning rate for `step_size`
    steps from `lr_min` to `lr_max`, then linearly decreasing the learning rate
    for another `step_size` steps from `lr_max` to (approximately) 0. Note that
    the scheduler is unaware of the total number of planned training steps, so
    be sure to pick `step_size` and set `global_step` accordingly.

    Args:
        lr_min: minimum (and starting) learning rate.
        lr_max: maximum learning rate.
        step_size: number of optimization steps in half a cycle.
        global_step: Tensorflow variable keeping track of the current
            optimization step.
        name: name to give the learning rate calculation operation.
    """
    with ops.name_scope(
        name, 'cyclic_lr', [lr_min, lr_max, step_size, global_step]
    ) as name:
        lr_min = tf.to_float(lr_min)
        lr_max = tf.to_float(lr_max)
        step_size = tf.to_float(step_size)
        def cyclic_lr():
            gs = tf.to_float(global_step) - 1
            cycle = 1. + gs//(2.*step_size)
            x = gs/step_size - 2.*cycle + 1.
            x_abs = tf.abs(x)
            # Decay the downward half towards 0 instead of lr_min
            _lr_min = tf.where(x < 0.0, lr_min, 0.0)
            lr_range = lr_max - _lr_min
            lr = _lr_min + lr_range*tf.maximum(0.0, 1. - x_abs)
            return lr
        return cyclic_lr


class CyclicSchedule:
    """Object implementation of a cyclic learning rate scheduler. The learning
    rate scheduling itself is identical to that used by `cyclic_lr_scheduler`,
    except this implements a callable class that also stores the hyperparameters
    of the schedule as class attributes. This object-oriented implementation is
    also a bit cleaner than returning a function handle.

    Args:
        lr_min: minimum (and starting) learning rate.
        lr_max: maximum learning rate.
        step_size: number of optimization steps in half a cycle.
        global_step: Tensorflow variable keeping track of the current
            optimization step.
    """
    def __init__(self, vmin, vmax, step_size, global_step=None):
        self.vmin = vmin
        self.vmax = vmax
        self.step_size = step_size
        if global_step is None:
            global_step = tf.train.get_or_create_global_step()
        self.global_step = global_step
        self.value = tf.Variable(vmin, trainable=False)

    def __call__(self):
        step = self.global_step.numpy() - 1
        cycle = 1. + step // (2. * self.step_size)
        x = step/self.step_size - 2. * cycle + 1.
        x_abs = tf.abs(x)
        # Decay the downward half towards 0 instead of vmin
        vmin = tf.where(x < 0.0, self.vmin, 0.0)
        vrange = self.vmax - vmin
        value = vmin + vrange * tf.cast(tf.maximum(0.0, 1. - x_abs), tf.float32)
        self.value.assign(value)
        return value
