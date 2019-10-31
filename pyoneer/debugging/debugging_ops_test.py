from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from pyoneer.debugging.debugging_ops import (Stopwatch, mock_spec)


class DebuggingTest(tf.test.TestCase):

    def test_stopwatch(self):
        with Stopwatch() as stopwatch:
            pass
        self.assertIsNotNone(stopwatch.start_time)
        self.assertIsNotNone(stopwatch.end_time)
        self.assertIsNotNone(stopwatch.duration)

    def test_mock_spec(self):
        mock_args = ([tf.zeros, tf.ones],
                     [tf.dtypes.float32, tf.dtypes.int32],
                     [[], [5], [10, 1], [100, 3, 6]],
                     [[2], [], [4, 6], [9, 7, 7, 4]])
        for args in zip(*mock_args):
            initializer, dtype, batch_shape, tensor_shape = args
            mocked_tensors = mock_spec(
                (tf.TensorShape(batch_shape), tf.TensorShape(batch_shape)),
                (tf.TensorSpec(shape=tensor_shape, dtype=dtype),
                 tf.TensorSpec(shape=tensor_shape, dtype=dtype)),
                (initializer, initializer))
            self.assertAllEqual(mocked_tensors,
                                (initializer(batch_shape + tensor_shape,
                                             dtype),
                                 initializer(batch_shape + tensor_shape,
                                             dtype)))


if __name__ == "__main__":
    tf.test.main()
