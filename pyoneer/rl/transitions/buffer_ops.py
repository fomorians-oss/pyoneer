from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading

import tensorflow as tf

from pyoneer.debugging import debugging_ops


def _variable_initializers(initializers, **kwargs):
    def wrap_initializer(initializer):
        def initializer_fn(shape, dtype):
            initial_value = initializer(shape, dtype)
            return tf.Variable(initial_value=initial_value, **kwargs)
        return initializer_fn

    if tf.nest.is_nested(initializers):
        return tf.nest.map_structure(wrap_initializer, initializers)
    return wrap_initializer(initializers)


class NstepBuffer(tf.Module):

    def __init__(self, specs, n_step, size, initializers=tf.zeros):
        """Stores n-step trajectories into a buffer.

        Args:
            specs: The nested `tf.TensorSpec`s to store in the buffer.
            n_step: The number of steps per trajectory.
            size: The total number of n_step trajectories of the buffer.
            initializers: The specs buffer initializers.
        """
        super(NstepBuffer, self).__init__(name='NstepBuffer')
        self._specs = specs
        self._n_step = tf.convert_to_tensor(n_step, tf.dtypes.int64)
        self._size = tf.convert_to_tensor(size, tf.dtypes.int64)
        self._trajectories = debugging_ops.mock_spec(
            tf.TensorShape([self._size, self._n_step]), specs,
            initializers=_variable_initializers(initializers, trainable=False))
        self._trajectory_slots_to_ids = tf.Variable(
            tf.fill([self._size], tf.cast(-1, tf.int64)))
        self._trajectory_slots_to_pos = tf.Variable(
            tf.fill([self._size], tf.cast(-1, tf.int64)))

    @property
    def n_step(self):
        return self._n_step

    @property
    def size(self):
        return self._size

    def _write(self, trajectory_slots, trajectory_ids, trajectory_pos,
               trajectories):
        """Writes n-step trajectories by ids.

        Args:
            trajectory_slots: The trajectory slots in memory. Tensor with
                shape [Batch].
            trajectory_ids: The trajectory ids corresponding to the episodes.
                Tensor with shape [Batch].
            trajectory_pos: The trajectory position in the episodes. Tensor
                with shape [Batch].
            trajectories: The nested structure of trajectories. Tensors with
                shape [Batch x N_Step x ...]
        """
        tf.debugging.assert_less(trajectory_slots, self._size)
        tf.debugging.assert_equal(tf.shape(trajectory_slots),
                                  tf.shape(trajectory_ids))
        tf.debugging.assert_equal(tf.shape(trajectory_slots),
                                  tf.shape(trajectory_pos))

        def scatter_trajectories(buffer, trajectory):
            buffer.scatter_nd_update(tf.expand_dims(trajectory_slots, axis=1),
                                     trajectory)

        # Populate the buffer with the trajectory.
        tf.nest.map_structure(scatter_trajectories, self._trajectories,
                              trajectories)

        # Populate the slot with the trajectory id.
        self._trajectory_slots_to_ids.scatter_nd_update(
            tf.expand_dims(trajectory_slots, axis=1), trajectory_ids)
        self._trajectory_slots_to_pos.scatter_nd_update(
            tf.expand_dims(trajectory_slots, axis=1), trajectory_pos)

    def _read_by_slots(self, trajectory_slots):
        """Read trajectories by slots.

        Args:
            trajectory_slots: The trajectory slots to read.
        """
        def gather_trajectories(buffer):
            return tf.gather(buffer, trajectory_slots)
        return tf.nest.map_structure(gather_trajectories, self._trajectories)

    def _read_by_ids(self, trajectory_ids, sort_by_pos=False,
                    finalize_slots_fn=None):
        """Read trajectories by ids.

        Args:
            trajectory_ids: The trajectory ids to read with shape [Batch].
            sort_by_pos: Sort the trajectory slots by position.
            finalize_slots_fn: Before returning slots, pass them to this
                function.

        Returns:
            The nested structure of tensors with shape
                [Batch x T x N_Step x ...].
        """
        def lookup_slots(slot_id):
            # Lookup the slots by id.
            id_mask = tf.equal(self._trajectory_slots_to_ids, slot_id)
            trajectory_slots = tf.squeeze(tf.where(id_mask), axis=1)

            # Sort the id-slots
            if sort_by_pos:
                trajectory_pos = tf.gather(self._trajectory_slots_to_pos,
                                           trajectory_slots)
                sort_ids = tf.argsort(trajectory_pos)
                trajectory_slots = tf.gather(trajectory_slots, sort_ids)

            # Finalize slots before returning them.
            if finalize_slots_fn is not None:
                trajectory_slots = finalize_slots_fn(trajectory_slots)
            return trajectory_slots

        trajectory_slots = tf.map_fn(lookup_slots, trajectory_ids)

        def gather_trajectories(buffer):
            return tf.gather(buffer, trajectory_slots)

        return tf.nest.map_structure(gather_trajectories, self._trajectories)


class ReplayBuffer(NstepBuffer):

    def __init__(self, specs, n_step, max_size, initializers=tf.zeros):
        super(ReplayBuffer, self).__init__(
            specs, n_step, max_size, initializers)
        self._id = tf.Variable(0, dtype=tf.dtypes.int64, trainable=False)
        self._count = tf.Variable(0, dtype=tf.dtypes.int64, trainable=False)

    @property
    def count(self):
        return tf.minimum(self._count, self.size)

    def write(self, trajectories, indices=None):
        # Check the shape of the trajectory.
        trajectory_shape = tf.cast(
            tf.shape(tf.nest.flatten(trajectories)[0]),
            tf.dtypes.int64)
        count, n_step = trajectory_shape[0], trajectory_shape[1]
        tf.debugging.assert_equal(n_step, self.n_step)

        # Compute the slots, ids, and positions.
        if indices is None:
            start = tf.identity(self._count)
            self._count.assign_add(count)
            end = tf.identity(self._count)
            indices = tf.range(start, end)
            indices = indices % self.size
        else:
            self._count.assign_add(count)

        trajectory_ids = tf.tile(
            tf.expand_dims(self._id, axis=0),
            [count])

        # Write the values to memory.
        self._write(indices, trajectory_ids, trajectory_ids, trajectories)

        # Increment the id by one.
        self._id.assign_add(1)

    def read(self, indices):
        return self._read_by_slots(indices)

    def _convert_slice_to_indices(self, start=None, stop=None, step=None):
        if start is None:
            start = 0
        start = tf.convert_to_tensor(start, dtype=tf.dtypes.int64)
        if stop is None:
            stop = -1
        stop = tf.convert_to_tensor(stop, dtype=tf.dtypes.int64)
        if step is None:
            step = 1
        step = tf.convert_to_tensor(step, dtype=tf.dtypes.int64)
        tf.debugging.assert_equal(tf.equal(step, 0), False, 'Slice step cannot be zero.')

        if start != abs(start):
            start = self.count + start + 1
        if stop != abs(stop):
            stop = self.count + stop + 1

        return tf.range(start, stop, step)

    def __getitem__(self, getter):
        if isinstance(getter, slice):
            indices = self._convert_slice_to_indices(
                getter.start, getter.stop, getter.step)
            return self.read(indices)
        getter = tf.convert_to_tensor(getter, dtype=tf.dtypes.int64)
        if getter.shape.rank > 0:
            return self.read(getter)
        values = self.read(tf.expand_dims(getter, axis=0))
        return tf.nest.map_structure(
            lambda t: tf.squeeze(t, axis=0),
            values)

    def __setitem__(self, setter, values):
        if isinstance(setter, slice):
            indices = self._convert_slice_to_indices(
                setter.start, setter.stop, setter.step)
            self.write(values, indices)
            return

        setter = tf.convert_to_tensor(setter, dtype=tf.dtypes.int64)
        if setter.shape.rank > 0:
            self.write(values, setter)
            return

        self.write(
            tf.nest.map_structure(
                lambda t: tf.expand_dims(t, axis=0), values),
            tf.expand_dims(setter, axis=0))
