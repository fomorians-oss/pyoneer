from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.eager as tfe

from copy import deepcopy
from tensorflow.python.eager import context
from tensorflow.python.platform import test

from pyoneer import losses

class GetL2LossTest(test.TestCase):
    def test_get_l2_loss(self):
        with context.eager_mode():
            variables = [
                tfe.Variable(tf.random.uniform((2, 3)), trainable=True),
                tfe.Variable(tf.random.uniform((3, 4)), trainable=True)
            ]
            l2_scale = 0.5
            l2_loss = float(losses.get_l2_loss(l2_scale, variables))

            correct_l2_loss = l2_scale * float(tf.reduce_sum([
                tf.reduce_sum(v**2)/2.0 for v in variables
            ]))
            self.assertAllEqual(correct_l2_loss, l2_loss)


if __name__ == '__main__':
    test.main()
