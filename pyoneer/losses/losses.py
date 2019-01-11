from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def l2_regularization(trainable_variables, scale=1.0):
    loss = scale * tf.add_n(
        [tf.nn.l2_loss(tvar) for tvar in trainable_variables])
    return loss
