from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from pyoneer.variables import variables_ops


class VariablesOpsTest(tf.test.TestCase):
    def test_update_target_variables(self):
        source = tf.Variable(1.0)
        target = tf.Variable(0.0)
        variables_ops.update_target_variables(
            source_variables=[source], target_variables=[target]
        )
        expected = tf.constant(1.0)
        self.assertAllEqual(target.numpy(), expected)

        source = tf.Variable(1.0)
        target = tf.Variable(0.0)
        variables_ops.update_target_variables(
            source_variables=[source], target_variables=[target], rate=0.5
        )
        expected = tf.constant(0.5)
        self.assertAllEqual(target.numpy(), expected)


if __name__ == "__main__":
    tf.test.main()
