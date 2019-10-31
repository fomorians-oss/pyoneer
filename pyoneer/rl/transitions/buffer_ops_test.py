from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

import tensorflow as tf

from pyoneer.debugging import debugging_ops
from pyoneer.rl.transitions import buffer_ops


class NstepBufferTest(tf.test.TestCase):

    def testReadBySlots(self):
        specs = (tf.TensorSpec([5], tf.dtypes.float32),)
        n_step = 1
        size = 1
        buffer = buffer_ops.NstepBuffer(specs, n_step, size)

        slots = tf.zeros([1], tf.int64)
        values = buffer._read_by_slots(slots)

        tf.nest.map_structure(
            self.assertAllEqual,
            values,
            debugging_ops.mock_spec(tf.TensorShape([1, n_step]), specs,
                                    initializers=tf.zeros))

    def testWriteReadBySlots(self):
        specs = (tf.TensorSpec([5], tf.dtypes.float32),)
        n_step = 1
        size = 1
        buffer = buffer_ops.NstepBuffer(specs, n_step, size)

        slots = tf.zeros([1], tf.int64)
        ids = tf.zeros([1], tf.int64)
        pos = tf.zeros([1], tf.int64)
        values = debugging_ops.mock_spec(tf.TensorShape([1, n_step]), specs,
                                         initializers=tf.ones)
        buffer._write(slots, ids, pos, values)
        actual_values = buffer._read_by_slots(slots)
        tf.nest.map_structure(self.assertAllEqual, values, actual_values)

    def testWriteReadByIds(self):
        specs = (tf.TensorSpec([5], tf.dtypes.float32),)
        n_step = 1
        size = 2
        buffer = buffer_ops.NstepBuffer(specs, n_step, size)

        slots = tf.constant([0, 1], tf.int64)
        ids = tf.constant([0, 0], tf.int64)
        pos = tf.constant([0, 1], tf.int64)
        values = debugging_ops.mock_spec(tf.TensorShape([2, n_step]), specs,
                                         initializers=tf.ones)

        buffer._write(slots, ids, pos, values)

        value_id = tf.constant([0], tf.int64)
        actual_values = buffer._read_by_ids(value_id)
        actual_values = tf.nest.map_structure(lambda x: tf.squeeze(x, axis=0),
                                              actual_values)
        tf.nest.map_structure(self.assertAllEqual, values, actual_values)

    def testWriteReadByIdsSorted(self):
        specs = (tf.TensorSpec([5], tf.dtypes.float32),)
        n_step = 1
        size = 2
        buffer = buffer_ops.NstepBuffer(specs, n_step, size)

        slots = tf.constant([0, 1], tf.int64)
        ids = tf.constant([0, 0], tf.int64)
        pos = tf.constant([0, 1], tf.int64)
        values_0 = debugging_ops.mock_spec(tf.TensorShape([1, n_step]), specs,
                                           initializers=tf.zeros)
        values_1 = debugging_ops.mock_spec(tf.TensorShape([1, n_step]), specs,
                                           initializers=tf.ones)

        values = tf.nest.map_structure(lambda x, y: tf.concat([x, y], axis=0),
                                       values_0, values_1)

        buffer._write(slots, ids, pos, values)

        value_id = tf.constant([0], tf.int64)
        actual_values = buffer._read_by_ids(value_id, sort_by_pos=True)

        actual_values = tf.nest.map_structure(lambda x: tf.squeeze(x, axis=0),
                                              actual_values)
        tf.nest.map_structure(self.assertAllEqual, values, actual_values)


if __name__ == "__main__":
    tf.test.main()
