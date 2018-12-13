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
            tf.set_random_seed(20)
            k = 15
            target_x = tf.random.uniform((3, k), minval=-1, maxval=1)
            target_y = tf.random.uniform((k, 2), minval=-1, maxval=1)
            target_matrix = target_x @ target_y

            x, y = linalg.funk_svd_solve(
                matrix=target_matrix, k=k, lr=5e-1, l2_scale=0, max_epochs=2000
            )
            matrix = x @ y

            self.assertAllClose(target_matrix.numpy(), matrix.numpy())


if __name__ == '__main__':
    test.main()
