from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.platform import test

from pyoneer.manip import array_ops


class ArrayOpsTest(tf.test.TestCase):
    def test_flatten(self):
        inputs = tf.constant([[0.0, 1.0, 2.0]])
        outputs = array_ops.flatten(inputs)
        expected = tf.constant([0.0, 1.0, 2.0])
        self.assertAllEqual(outputs, expected)

    def test_pad_or_truncate(self):
        inputs = [["apple", "is", "red"], ["i", "dont", "like", "apples"]]

        # truncate
        outputs = tf.stack(
            [
                array_ops.pad_or_truncate(input, sizes=[3], constant_values="<PAD>")
                for input in inputs
            ],
            axis=0,
        )
        expected = tf.constant([["apple", "is", "red"], ["i", "dont", "like"]])
        self.assertAllEqual(outputs, expected)

        # pad
        outputs = tf.stack(
            [
                array_ops.pad_or_truncate(input, sizes=[4], constant_values="<PAD>")
                for input in inputs
            ],
            axis=0,
        )
        expected = tf.constant(
            [["apple", "is", "red", "<PAD>"], ["i", "dont", "like", "apples"]]
        )
        self.assertAllEqual(outputs, expected)

    def test_shift(self):
        # shift forwards
        inputs = tf.constant([[0.0, 1.0, 2.0]])
        outputs = array_ops.shift(inputs, shift=1, axis=1)
        expected = tf.constant([[0.0, 0.0, 1.0]])
        self.assertAllEqual(outputs, expected)

        # shift backwards
        inputs = tf.constant([[0.0, 1.0, 2.0]])
        outputs = array_ops.shift(inputs, shift=-1, axis=1)
        expected = tf.constant([[1.0, 2.0, 0.0]])
        self.assertAllEqual(outputs, expected)

        # shift forwards fill
        inputs = tf.constant([[0.0, 1.0, 2.0]])
        outputs = array_ops.shift(inputs, shift=1, axis=1, padding_values=-1.0)
        expected = tf.constant([[-1.0, 0.0, 1.0]])
        self.assertAllEqual(outputs, expected)

        # shift backwards fill
        inputs = tf.constant([[0.0, 1.0, 2.0]])
        outputs = array_ops.shift(inputs, shift=-1, axis=1, padding_values=3.0)
        expected = tf.constant([[1.0, 2.0, 3.0]])
        self.assertAllEqual(outputs, expected)

        # shift forwards more
        inputs = tf.constant([[0.0, 1.0, 2.0]])
        outputs = array_ops.shift(inputs, shift=2, axis=1)
        expected = tf.constant([[0.0, 0.0, 0.0]])
        self.assertAllEqual(outputs, expected)

        # shift backwards more
        inputs = tf.constant([[0.0, 1.0, 2.0]])
        outputs = array_ops.shift(inputs, shift=-2, axis=1)
        expected = tf.constant([[2.0, 0.0, 0.0]])
        self.assertAllEqual(outputs, expected)

        # shift using tensor constants
        values = tf.constant([[0.0, 0.0, 1.0]])
        bootstrap_values = tf.constant([1.0])
        expected_values_next = tf.concat(
            [values[:, 1:], bootstrap_values[:, None]], axis=1
        )
        actual_values_next = array_ops.shift(
            values, shift=-1, axis=1, padding_values=bootstrap_values[:, None]
        )
        self.assertAllEqual(actual_values_next, expected_values_next)


if __name__ == "__main__":
    test.main()
