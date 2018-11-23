from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.eager import context
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test

from pyoneer.math import logical_ops


class LogicalOpsTest(test.TestCase):

    def test_isclose(self):
        with context.eager_mode():
            x = array_ops.constant([[0.9, 1.0, 1.1, 1.2]])
            actual_x = logical_ops.isclose(x, 1., epsilon=.1)
            expected_x = array_ops.constant([[True, True, True, False]])
            self.assertAllEqual(actual_x, expected_x)


if __name__ == '__main__':
    test.main()