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


class _TestTuple(collections.namedtuple('_TestTuple', ['a', 'b'])):
    __slots__ = ()


class ReplayBufferTest(tf.test.TestCase):

    def testWriteReadDict(self):
        specs = dict(a=tf.TensorSpec([5], tf.dtypes.float32),
                     b=tf.TensorSpec([10, 23], tf.dtypes.float32))
        n_step = 1
        size = 20
        buffer = buffer_ops.ReplayBuffer(specs, n_step, size)
        checkpoint = tf.train.Checkpoint(buffer=buffer)

        expected_values = debugging_ops.mock_spec(
            tf.TensorShape([10, n_step]), specs,
            initializers=tf.keras.initializers.RandomUniform(-1., 1.))

        # Test write, slice to half-size.
        buffer.write(expected_values)
        values = buffer[:-1]

        tf.nest.map_structure(
            self.assertAllEqual,
            values,
            expected_values)
        save_path = checkpoint.save(os.path.join(self.get_temp_dir(), "ckpt"))

        # Test gather.
        indices = [0, 2, 4, 8]
        expected_gather_values = tf.nest.map_structure(
            lambda x: tf.gather(x, indices),
            expected_values)

        gather_values = buffer[indices]

        tf.nest.map_structure(
            self.assertAllEqual,
            gather_values,
            expected_gather_values)

        # Read single value.
        index = 0
        expected_value = tf.nest.map_structure(
            lambda x: x[index],
            expected_values)

        value = buffer[index]

        tf.nest.map_structure(
            self.assertAllEqual,
            value,
            expected_value)

        # Write to full size.
        buffer.write(expected_values)
        values = buffer[:-1]

        tf.nest.map_structure(
            self.assertAllEqual,
            values,
            tf.nest.map_structure(
                lambda x: tf.concat([x, x], axis=0),
                expected_values))

        # Restore the old buffer, which has half size.
        checkpoint.restore(save_path)
        values = buffer[:-1]

        tf.nest.map_structure(
            self.assertAllEqual,
            values,
            expected_values)

    def testWriteReadTuple(self):
        specs = (tf.TensorSpec([5], tf.dtypes.float32),
                 tf.TensorSpec([10, 23], tf.dtypes.float32))
        n_step = 1
        size = 20
        buffer = buffer_ops.ReplayBuffer(specs, n_step, size)
        checkpoint = tf.train.Checkpoint(buffer=buffer)

        expected_values = debugging_ops.mock_spec(
            tf.TensorShape([10, n_step]), specs,
            initializers=tf.keras.initializers.RandomUniform(-1., 1.))

        # Test write, slice to half-size.
        buffer.write(expected_values)
        values = buffer[:-1]

        tf.nest.map_structure(
            self.assertAllEqual,
            values,
            expected_values)
        save_path = checkpoint.save(os.path.join(self.get_temp_dir(), "ckpt"))

        # Test gather.
        indices = [0, 2, 4, 8]
        expected_gather_values = tf.nest.map_structure(
            lambda x: tf.gather(x, indices),
            expected_values)

        gather_values = buffer[indices]

        tf.nest.map_structure(
            self.assertAllEqual,
            gather_values,
            expected_gather_values)

        # Read single value.
        index = 0
        expected_value = tf.nest.map_structure(
            lambda x: x[index],
            expected_values)

        value = buffer[index]

        tf.nest.map_structure(
            self.assertAllEqual,
            value,
            expected_value)

        # Write to full size.
        buffer.write(expected_values)
        values = buffer[:-1]

        tf.nest.map_structure(
            self.assertAllEqual,
            values,
            tf.nest.map_structure(
                lambda x: tf.concat([x, x], axis=0),
                expected_values))

        # Restore the old buffer, which has half size.
        checkpoint.restore(save_path)
        values = buffer[:-1]

        tf.nest.map_structure(
            self.assertAllEqual,
            values,
            expected_values)

    def testWriteReadNamedTuple(self):
        specs = _TestTuple(a=tf.TensorSpec([5], tf.dtypes.float32),
                           b=tf.TensorSpec([10, 23], tf.dtypes.float32))
        n_step = 1
        size = 20
        buffer = buffer_ops.ReplayBuffer(specs, n_step, size)
        checkpoint = tf.train.Checkpoint(buffer=buffer)

        expected_values = debugging_ops.mock_spec(
            tf.TensorShape([10, n_step]), specs,
            initializers=tf.keras.initializers.RandomUniform(-1., 1.))

        # Test write, slice to half-size.
        buffer.write(expected_values)
        values = buffer[:-1]

        tf.nest.map_structure(
            self.assertAllEqual,
            values,
            expected_values)
        save_path = checkpoint.save(os.path.join(self.get_temp_dir(), "ckpt"))

        # Test gather.
        indices = [0, 2, 4, 8]
        expected_gather_values = tf.nest.map_structure(
            lambda x: tf.gather(x, indices),
            expected_values)

        gather_values = buffer[indices]

        tf.nest.map_structure(
            self.assertAllEqual,
            gather_values,
            expected_gather_values)

        # Read single value.
        index = 0
        expected_value = tf.nest.map_structure(
            lambda x: x[index],
            expected_values)

        value = buffer[index]

        tf.nest.map_structure(
            self.assertAllEqual,
            value,
            expected_value)

        # Write to full size.
        buffer.write(expected_values)
        values = buffer[:-1]

        tf.nest.map_structure(
            self.assertAllEqual,
            values,
            tf.nest.map_structure(
                lambda x: tf.concat([x, x], axis=0),
                expected_values))

        # Restore the old buffer, which has half size.
        checkpoint.restore(save_path)
        values = buffer[:-1]

        tf.nest.map_structure(
            self.assertAllEqual,
            values,
            expected_values)


if __name__ == "__main__":
    tf.test.main()
