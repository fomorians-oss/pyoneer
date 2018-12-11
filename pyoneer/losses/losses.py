from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def get_l2_loss(l2_scale, trainable_variables):
    l2_loss = l2_scale * tf.add_n([
        tf.nn.l2_loss(tvar) for tvar in trainable_variables
    ])
    return l2_loss
