from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import test

from pyoneer.nn.rnn_cell import fast_weights_impl


class FastWeightsTest(test.TestCase):
    def test_FastWeightsRNNCell(self):
        with context.eager_mode():
            inputs = array_ops.zeros([1, 2])
            initial_states = (array_ops.zeros([1, 2]),
                              array_ops.zeros([1, 2, 2]))
            with variable_scope.variable_scope(
                    'test_fast_weights_cell',
                    initializer=init_ops.constant_initializer(0.5)):
                cell = fast_weights_impl.FastWeightsRNNCell(2)
                outputs, states = cell(inputs, initial_states)
            self.assertAllClose(outputs, array_ops.constant([[0., 0.]]))


if __name__ == '__main__':
    test.main()