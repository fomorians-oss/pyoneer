from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from pyoneer.variables import variable_ops


class VariableOpsTest(tf.test.TestCase):

    def testUpdateVariables(self):
        for rate in [1., .75, .5, .25, 0.1]:
            target = [tf.Variable(2., dtype=tf.dtypes.float32)]
            source = [tf.Variable(1., dtype=tf.dtypes.float32)]
            expected_target = [rate * source[0] + (1. - rate) * target[0]]
            variable_ops.update_variables(source, target, rate=rate)
            tf.nest.map_structure(
                lambda x, y: self.assertAllEqual(x.numpy(), y.numpy()),
                expected_target, target)


if __name__ == "__main__":
    tf.test.main()