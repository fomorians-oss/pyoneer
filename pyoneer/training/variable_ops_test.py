import tensorflow.contrib.eager as tfe

from tensorflow.python.eager import context
from tensorflow.python.platform import test

from pyoneer.training import variable_ops


class VariableOpsTest(test.TestCase):
    def test_update_variables_copy(self):
        with context.eager_mode():
            a = tfe.Variable([0.0, 0.1, 0.5, 1.0], trainable=False)
            b = tfe.Variable([1.0, 0.5, 0.1, 0.0], trainable=False)
            variable_ops.update_variables([a], [b], rate=1.0)
            expected = tfe.Variable([0.0, 0.1, 0.5, 1.0], trainable=False)
            self.assertAllEqual(a.numpy(), expected.numpy())
            self.assertAllEqual(b.numpy(), expected.numpy())

    def test_update_variables_interpolate(self):
        with context.eager_mode():
            a = tfe.Variable([0.0, 0.1, 0.5, 1.0], trainable=False)
            b = tfe.Variable([1.0, 0.5, 0.1, 0.0], trainable=False)
            variable_ops.update_variables([a], [b], rate=0.5)
            expected_a = tfe.Variable([0.0, 0.1, 0.5, 1.0], trainable=False)
            expected_b = tfe.Variable([0.5, 0.3, 0.3, 0.5], trainable=False)
            self.assertAllEqual(a.numpy(), expected_a.numpy())
            self.assertAllEqual(b.numpy(), expected_b.numpy())


if __name__ == '__main__':
    test.main()
