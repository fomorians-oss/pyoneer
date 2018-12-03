from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.framework import ops

def cyclic_lr_scheduler(lr_min, lr_max, step_size, global_step, name=None):
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
    def __init__(self, vmin, vmax, step_size, half=False, global_step=None):
        self.vmin = vmin
        self.vmax = vmax
        self.step_size = step_size
        self.half = half  # What is this supposed to do??
        if global_step is None:
            global_step = tf.train.get_or_create_global_step()
        self.global_step = global_step

    def __call__(self):
        step = self.global_step.numpy() - 1
        cycle = 1. + step//(2.*self.step_size)
        x = step/self.step_size - 2.*cycle + 1.
        x_abs = tf.abs(x)
        # Decay the downward half towards 0 instead of vmin
        vmin = tf.where(x < 0.0, self.vmin, 0.0)
        vrange = self.vmax - vmin
        value = vmin + vrange*tf.cast(tf.maximum(0.0, 1. - x_abs), tf.float32)
        return value
