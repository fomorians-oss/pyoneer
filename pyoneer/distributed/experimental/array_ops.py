from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import collections
import tensorflow as tf

from pyoneer.debugging import debugging_ops
from pyoneer.distributed import distributed_ops


class Array(distributed_ops.TensorCodec):

    def __init__(self, size, specs, name=None, pipe=None):
        """Creates a new Array.

        Args:
            pipe: The redis server.
            name: The redis key for the queue. Each name is unique to
                the corresponding shared memory.
            specs: The possibly nested structure containing `tf.TensorSpec`s.
        """
        dtypes = tf.nest.map_structure(lambda s: s.dtype, specs)
        super(Array, self).__init__(dtypes)
        self._size = size
        self._specs = specs
        self._key = distributed_ops.assign_key_name(name, 'Array')
        if pipe is None:
            pipe = distributed_ops.get_default_pipe()
            assert pipe is not None, ('No default pipe set, must use `set_default_pipe`'
                                      'or pass a pipe that is not `None`.')
        self._pipe = pipe

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

        length = self.len()
        if start != abs(start):
            start = length + tf.cast(start + 1, tf.dtypes.int64)
        if stop != abs(stop):
            stop = length + tf.cast(stop + 1, tf.dtypes.int64)

        return tf.range(start, stop, step)

    def __getitem__(self, getter):
        if isinstance(getter, slice):
            indices = self._convert_slice_to_indices(
                getter.start, getter.stop, getter.step)
            return self.gather(indices)

        getter = tf.convert_to_tensor(getter, dtype=tf.dtypes.int64)
        if getter.shape.rank > 0:
            return self.gather(getter)

        return self.get(getter)

    def __setitem__(self, setter, values):
        if isinstance(setter, slice):
            indices = self._convert_slice_to_indices(
                setter.start, setter.stop, setter.step)
            return self.scatter(indices, values)

        setter = tf.convert_to_tensor(setter, dtype=tf.dtypes.int64)
        if setter.shape.rank > 0:
            self.scatter(setter, values)
            return

        self.set(setter, values)

    def __len__(self):
        return self.size

    @property
    def specs(self):
        return self._specs

    def _get_fn(self, index):
        # TODO(wenkesj): Return default value for values that don't exist.
        item = self._pipe.hget(self._key, index.item())
        if item:
            item = item[1]
        return item

    @tf.function
    def get(self, index):
        """Get a nested structure.

        Args:
            index: The index to get the value.

        Returns:
            The nested structure.
        """
        item = tf.numpy_function(self._get_fn, (index,), tf.dtypes.string)
        item = tf.ensure_shape(item, [])
        decoded_item = self.decode(item)
        return decoded_item

    def _gather_fn(self, indices):
        # TODO(wenkesj): Return default value for values that don't exist.
        items = self._pipe.hmget(self._key, *indices.astype(dtype='|S1'))
        if items:
            return np.stack(items, axis=0), len(items)
        return np.asarray([b'']), 0

    @tf.function
    def gather(self, indices):
        """Gather nested structures.

        Args:
            indices: The indices to gather values.

        Returns:
            The nested structure.
        """
        items, num_items = tf.numpy_function(self._gather_fn, (indices,),
                                             (tf.dtypes.string, tf.dtypes.int64))
        num_items = tf.ensure_shape(num_items, [])

        if tf.equal(num_items, 0):
            return debugging_ops.mock_spec(tf.TensorShape([0]), self.specs, tf.zeros)

        items.set_shape([None])

        decoded_items = tf.nest.map_structure(
            lambda x: x[None, ...],
            self.decode(items[0]))
        if tf.greater(num_items - 1, 0):
            for index in tf.range(num_items - 1):
                decoded_items = tf.nest.map_structure(
                    lambda x, y: tf.concat([x, y[None]], axis=0),
                    decoded_items,
                    self.decode(items[index + 1]))
        return decoded_items

    def _scatter_fn(self, indices, values):
        self._pipe.hmset(self._key, dict(zip(indices.astype(dtype='|S1'), values)))

    @tf.function
    def scatter(self, indices, values):
        """Scatter nested structures.

        Args:
            indices: The indices to scatter values.
            indices: Values to scatter.
        """
        encoded_values = tf.map_fn(self.encode, values, dtype=tf.dtypes.string)
        tf.numpy_function(self._scatter_fn, (indices, encoded_values), ())

    def _set_fn(self, index, value):
        self._pipe.hset(self._key, index.item(), value)

    @tf.function
    def set(self, index, structure):
        """Set a nested structure.

        Args:
            index: The index to set the value.
            structure: The nested structure.
        """
        buffer = self.encode(structure)
        tf.numpy_function(self._set_fn, (index, buffer), ())
