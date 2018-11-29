from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.eager import context
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test

from pyoneer.manip import array_ops as parray_ops


class ArrayOpsTest(test.TestCase):

    def test_pad_or_truncate(self):
        with context.eager_mode():
            x = array_ops.constant([[0, 1, 2]])
            actual_x = parray_ops.pad_or_truncate(x, maxsize=4, axis=1, pad_value=3)
            expected_x = array_ops.constant([[0, 1, 2, 3]])
            self.assertAllClose(actual_x, expected_x)

            x = array_ops.constant([[0, 1, 2, 3, 4]])
            actual_x = parray_ops.pad_or_truncate(x, maxsize=4, axis=1, pad_value=3)
            self.assertAllClose(actual_x, expected_x)


if __name__ == '__main__':
    test.main()