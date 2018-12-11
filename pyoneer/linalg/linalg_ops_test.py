from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.eager as tfe

from copy import deepcopy
from tensorflow.python.eager import context
from tensorflow.python.platform import test

import pyoneer.linalg as linalg

class FunkSVDSolveTest(test.TestCase):
    def test_funk_svd_solve(self):
        with context.eager_mode():
            tf.set_random_seed(0)
            k = 3
            target_x = tf.random.uniform((4, k), minval=-2, maxval=2)
            target_y = tf.random.uniform((2, k), minval=-2, maxval=2)
            matrix = target_x @ tf.transpose(target_y)
            l2_scale = 0
            tol = 1e-8
            step = tfe.Variable(0, trainable=False)
            optimizer = tf.train.AdamOptimizer(learning_rate=1e-6)

            x, y = linalg.funk_svd_solve(
                matrix, k, optimizer, step=step, l2_scale=l2_scale,
                tol=tol, max_epochs=1000, n_epochs_no_change=50
            )

            self.assertAllClose(
                (target_x.numpy(), target_y.numpy()), (x.numpy(), y.numpy())
            )



if __name__ == '__main__':
    test.main()
