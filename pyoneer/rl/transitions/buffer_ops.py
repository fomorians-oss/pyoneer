from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_probability as tfp

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
    def __init__(self, specs, n_step, max_size):
        """Stores n-step trajectories into a buffer.

        Args:
            specs: The nested `tf.TensorSpec`s to store in the buffer.
            n_step: The number of steps per trajectory.
            max_size: The total number of n_step trajectories of the buffer.
        """
        super(NstepBuffer, self).__init__(name="NstepBuffer")
        self._specs = specs
        self._n_step = tf.convert_to_tensor(n_step, tf.dtypes.int64)
        self._max_size = tf.convert_to_tensor(max_size, tf.dtypes.int64)

        trajectories = debugging_ops.mock_spec(
            tf.TensorShape([self._max_size, self._n_step]),
            specs,
            initializers=_variable_initializers(tf.zeros, trainable=False),
        )
        # Force the nested structure variables to be tracked.
        self._trajectories_flat = tf.nest.flatten(trajectories)
        self._trajectories = trajectories
        self._trajectory_slots_to_ids = tf.Variable(
            tf.fill([self._max_size], tf.cast(-1, tf.dtypes.int64)), trainable=False
        )
        self._trajectory_slots_to_pos = tf.Variable(
            tf.fill([self._max_size], tf.cast(-1, tf.dtypes.int64)), trainable=False
        )
        self._non_empty = tf.Variable(0, dtype=tf.dtypes.int64, trainable=False)

    @property
    def n_step(self):
        return self._n_step

    @property
    def max_size(self):
        return self._max_size

    @property
    def non_empty(self):
        return tf.identity(self._non_empty)

    def _write(self, trajectory_slots, trajectory_ids, trajectory_pos, trajectories):
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
        tf.debugging.assert_less(trajectory_slots, self._max_size)
        tf.debugging.assert_equal(tf.shape(trajectory_slots), tf.shape(trajectory_ids))
        tf.debugging.assert_equal(tf.shape(trajectory_slots), tf.shape(trajectory_pos))
        self._non_empty.assign(
            tf.minimum(
                self._non_empty
                + tf.cast(tf.shape(trajectory_slots)[0], tf.dtypes.int64),
                self._max_size,
            )
        )

        def scatter_trajectories(buffer, trajectory):
            buffer.scatter_nd_update(
                tf.expand_dims(trajectory_slots, axis=1), trajectory
            )

        # Populate the buffer with the trajectory.
        tf.nest.map_structure(scatter_trajectories, self._trajectories, trajectories)

        # Populate the slot with the trajectory id.
        self._trajectory_slots_to_ids.scatter_nd_update(
            tf.expand_dims(trajectory_slots, axis=1), trajectory_ids
        )
        self._trajectory_slots_to_pos.scatter_nd_update(
            tf.expand_dims(trajectory_slots, axis=1), trajectory_pos
        )

    def _read_by_slots(self, trajectory_slots):
        """Read trajectories by slots.

        Args:
            trajectory_slots: The trajectory slots to read.
        """

        def gather_trajectories(buffer):
            return tf.gather(buffer, trajectory_slots)

        return tf.nest.map_structure(gather_trajectories, self._trajectories)

    def _read_by_ids(self, trajectory_ids, sort_by_pos=False, finalize_slots_fn=None):
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
                trajectory_pos = tf.gather(
                    self._trajectory_slots_to_pos, trajectory_slots
                )
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


class NstepLRUBuffer(NstepBuffer):
    """Create a new NstepLRUBuffer.

    The `NstepLRUBuffer` is a buffer of tensors that returns a
    uniformly sampled dataset, and removes the least-recent
    entries when written and the max size is reached. The
    least-recent is computed by the `timestamp = episode_id + n_step_id`.
    """

    def write(self, episode_ids, n_step_ids, trajectories):
        """Write trajectoies into the buffer."""
        size = tf.shape(episode_ids)[0]
        old_time_stamps = self._trajectory_slots_to_ids + self._trajectory_slots_to_pos
        slots_sorted = tf.argsort(old_time_stamps)
        slots = tf.cast(slots_sorted[:size], tf.dtypes.int64)
        self._write(
            slots,
            tf.cast(episode_ids, tf.dtypes.int64),
            tf.cast(n_step_ids, tf.dtypes.int64),
            trajectories,
        )

    def sample_dataset(self, num_samples, replace=True):
        """Sample a dataset from the buffer."""
        num_non_empty = self.non_empty
        tf.debugging.assert_less(
            tf.cast(num_samples, tf.dtypes.int64),
            tf.cast(num_non_empty + 1, tf.dtypes.int64),
        )

        if replace:
            sample_slots = tf.cast(
                tf.random.uniform(
                    [num_samples], 0.0, tf.cast(num_non_empty, tf.dtypes.float32)
                ),
                tf.dtypes.int64,
            )
        else:
            sample_slots = tf.cast(
                tf.random.shuffle(tf.range(num_non_empty))[:num_samples],
                tf.dtypes.int64,
            )

        samples = self._read_by_slots(sample_slots)
        sample_dataset = tf.data.Dataset.from_tensor_slices(samples)
        return sample_dataset


class NstepPrioritizedBuffer(NstepBuffer):
    def __init__(self, specs, n_step, alpha, beta, max_size, epsilon=1e-6):
        """Create a new NstepPrioritizedBuffer.

        The `NstepPrioritizedBuffer` is a buffer of tensors that returns a
        sampled dataset according to the priorities associated with the
        trajectories. It removes the least-priorities when written and the
        max size is reached.
        """
        super().__init__(specs, n_step, max_size)
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self._trajectory_slots_to_priorities = tf.Variable(
            tf.fill([self._max_size], tf.cast(0.0, tf.dtypes.float32)), trainable=False
        )
        self._priority_sum = tf.Variable(0.0, dtype=tf.dtypes.float32, trainable=False)

    @property
    def priority_sum(self):
        return tf.identity(self._priority_sum)

    def write(self, episode_ids, n_step_ids, priorities, trajectories):
        """Write trajectoies into the buffer."""
        priorities += self.epsilon
        size = tf.shape(episode_ids)[0]
        slots_sorted = tf.argsort(self._trajectory_slots_to_priorities)
        slots = tf.cast(slots_sorted[:size], tf.dtypes.int64)
        self._trajectory_slots_to_priorities.scatter_nd_update(
            tf.expand_dims(slots, axis=1), priorities
        )
        self._priority_sum.assign_add(
            tf.reduce_sum(self._trajectory_slots_to_priorities ** self.alpha)
        )
        self._write(
            slots,
            tf.cast(episode_ids, tf.dtypes.int64),
            tf.cast(n_step_ids, tf.dtypes.int64),
            trajectories,
        )

    def update_priorities(self, slots, priorities):
        """Update the trajectory priorities given the slot."""
        priorities += self.epsilon
        self._trajectory_slots_to_priorities.scatter_nd_update(
            tf.expand_dims(slots, axis=1), priorities
        )
        self._priority_sum.assign_add(
            tf.reduce_sum(self._trajectory_slots_to_priorities ** self.alpha)
        )

    def sample_dataset(self, num_samples):
        """Sample a dataset from the buffer."""
        num_non_empty = self.non_empty
        tf.debugging.assert_less(
            tf.cast(num_samples, tf.dtypes.int64),
            tf.cast(num_non_empty + 1, tf.dtypes.int64),
        )

        sample_mask = self._trajectory_slots_to_priorities > 0.0
        probabilities = (
            self._trajectory_slots_to_priorities ** self.alpha
        ) / self._priority_sum

        is_weights = tf.where(
            sample_mask,
            (tf.cast(num_non_empty, tf.dtypes.float32) * probabilities) ** -self.beta,
            tf.zeros_like(probabilities),
        )
        indices = tfp.distributions.Categorical(probs=probabilities).sample(num_samples)

        is_weights = tf.gather(is_weights, indices)
        max_is_weight = tf.reduce_max(is_weights)
        is_weights /= max_is_weight

        samples = self._read_by_slots(indices)
        sample_dataset = tf.data.Dataset.from_tensor_slices(
            (samples, is_weights, indices)
        )
        return sample_dataset
