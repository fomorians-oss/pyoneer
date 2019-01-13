from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import context
from tensorflow.python.platform import test


class ActivationOpsTest(test.TestCase):
    def test_swish(self):
        with context.eager_mode():
            self.assertAllEqual(False, True)


if __name__ == '__main__':
    test.main()
