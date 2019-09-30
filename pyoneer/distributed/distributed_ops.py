from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import collections
import tensorflow as tf

from pyoneer.debugging import debugging_ops


def _stack_nested(list_of_nested):
    stacked_items = tf.nest.map_structure(
        lambda x: x[None, ...], list_of_nested[0])
    for next_item in list_of_nested[1:]:
        stacked_items = tf.nest.map_structure(
            lambda x, y: tf.concat([x, y[None, ...]], axis=0),
            stacked_items,
            next_item)
    return stacked_items


class TensorCodec(object):

    def __init__(self, dtypes):
        """Creates a new TensorCodec.

        This creates a codec that encodes structures to strings
            and decodes messages back to the structures.

        Args:
            dtypes: The possibly nested structure containing tensor
                dtypes.
        """
        self._dtypes = dtypes
        self._length_dtype = tf.dtypes.int32
        self._count = len(tf.nest.flatten([self._dtypes]))
        length_bytes = tf.strings.bytes_split(
            tf.io.serialize_tensor(tf.zeros([self._count], self._length_dtype)))
        num_length_bytes = tf.shape(length_bytes)[0]
        self._offset = tf.cast(num_length_bytes, self._length_dtype)

    @property
    def dtypes(self):
        return self._dtypes

    @tf.function
    def encode(self, item):
        """Encode a structure into a packed message.

        Args:
            item: A structure with the same structure as `self.dtypes`.

        Returns:
            The encoded message.
        """
        # Ensure the same structure.
        tf.nest.assert_same_structure(item, self.dtypes)

        lengths = []
        structure_msg = []
        for tensor in tf.nest.flatten(item):
            msg = tf.io.serialize_tensor(tensor)
            structure_msg.append(msg)
            msg_bytes = tf.strings.bytes_split(msg)
            num_msg_bytes = tf.shape(msg_bytes)[0]
            lengths.append(tf.cast(num_msg_bytes, self._length_dtype))

        lengths_tensor = tf.stack(lengths, axis=0)
        lengths_msg = tf.io.serialize_tensor(lengths_tensor)

        buffer_list = tf.concat([lengths_msg[None], structure_msg], axis=0)
        buffer = tf.strings.reduce_join(buffer_list)
        return buffer

    @tf.function
    def decode(self, item):
        """Decode a packed message that represents a structure.

        Args:
            item: A string representing the encoded message.

        Returns:
            The decoded structure.
        """
        item_bytes = tf.strings.bytes_split(item)
        offset = tf.identity(self._offset)

        # Get the lengths of the items in the encoded string.
        lengths_msg = tf.strings.reduce_join(item_bytes[0:offset])
        lengths = tf.io.parse_tensor(lengths_msg, self._length_dtype)
        lengths.set_shape([self._count])

        # Decode the item into the same encoded structure.
        decoded_items = []
        lengths = tf.nest.pack_sequence_as(
            self._dtypes, tf.unstack(lengths, self._count, axis=0))

        lengths = tf.nest.flatten(lengths)
        dtypes = tf.nest.flatten(self._dtypes)
        for length, dtype in zip(lengths, dtypes):
            length.set_shape([])
            item_structure = tf.strings.reduce_join(item_bytes[offset:offset+length])
            offset = offset + length
            decoded_item = tf.io.parse_tensor(item_structure, dtype)
            decoded_items.append(decoded_item)

        structure = tf.nest.pack_sequence_as(
            self._dtypes, decoded_items)
        return structure


class Queue(TensorCodec):

    def __init__(self, pipe, key, dtypes):
        """Creates a new Queue.

        This creates a distributed queue datastructure.

        Args:
            pipe: The redis server.
            key: The redis key for the queue.
            dtypes: The possibly nested structure containing tensor
                dtypes.
        """
        super(Queue, self).__init__(dtypes)
        self._pipe = pipe
        self._key = key

    def _enqueue_fn(self, buffer):
        self._pipe.rpush(self._key, buffer)

    @tf.function
    def enqueue(self, structure):
        """Enqueue a nested structure.

        Args:
            structure: The nested structure.
        """
        buffer = self.encode(structure)
        tf.numpy_function(self._enqueue_fn, (buffer,), ())

    def _dequeue_fn(self):
        item = self._pipe.blpop(self._key)
        if item:
            item = item[1]
        return item

    @tf.function
    def dequeue(self):
        """Dequeue a nested structure.

        Returns:
            The nested structure.
        """
        item = tf.numpy_function(self._dequeue_fn, (), tf.dtypes.string)
        item = tf.ensure_shape(item, [])
        decoded_item = self.decode(item)
        return decoded_item

    def _dequeue_many_fn(self, many):
        pipeline = self._pipe.pipeline()
        pipeline.multi()
        for _ in range(many.item()):
            pipeline.blpop(self._key)
        items = pipeline.execute()
        _, items = zip(*items)
        return np.stack(items, axis=0)

    @tf.function
    def dequeue_many(self, many):
        tf.debugging.assert_greater_equal(many, 1, '`many` < 1.')
        items = tf.numpy_function(self._dequeue_many_fn, (many,), tf.dtypes.string)
        items.set_shape([None])

        decoded_items = tf.nest.map_structure(
            lambda x: x[None, ...],
            self.decode(items[0]))
        if tf.greater(many - 1, 0):
            for index in tf.range(many - 1):
                decoded_items = tf.nest.map_structure(
                    lambda x, y: tf.concat([x, y[None]], axis=0),
                    decoded_items,
                    self.decode(items[index]))
        return decoded_items


class List(TensorCodec):

    def __init__(self, pipe, key, specs):
        """Creates a new List.

        This creates a distributed list datastructure.

        Args:
            pipe: The redis server.
            key: The redis key for the queue.
            dtypes: The possibly nested structure containing tensor
                dtypes.
        """
        dtypes = tf.nest.map_structure(lambda s: s.dtype, specs)
        super(List, self).__init__(dtypes)
        self._specs = specs
        self._pipe = pipe
        self._key = key

    def __getitem__(self, getter):
        if isinstance(getter, slice):
            return self.slice_get(
                start=getter.start,
                stop=getter.stop,
                step=getter.step)
        getter = tf.convert_to_tensor(getter, dtype=tf.dtypes.int64)
        if getter.shape.rank > 0:
            return self.gather(getter)
        return self.get(getter)

    @property
    def specs(self):
        return self._specs

    def _append_fn(self, buffer):
        self._pipe.rpush(self._key, buffer)

    @tf.function
    def append(self, structure):
        """Append a nested structure.

        Args:
            structure: The nested structure.
        """
        buffer = self.encode(structure)
        tf.numpy_function(self._append_fn, (buffer,), ())

    def _pop_fn(self):
        item = self._pipe.brpop(self._key)
        if item:
            item = item[1]
        return item

    @tf.function
    def pop(self):
        """Pop a nested structure.

        Returns:
            The nested structure.
        """
        item = tf.numpy_function(self._pop_fn, (), tf.dtypes.string)
        item = tf.ensure_shape(item, [])
        decoded_item = self.decode(item)
        return decoded_item

    def _get_fn(self, index):
        item = self._pipe.lindex(self._key, index.item())
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
        pipeline = self._pipe.pipeline()
        pipeline.multi()
        for index in indices:
            pipeline.lindex(self._key, index.item())
        items = pipeline.execute()
        _, items = zip(*items)
        return np.stack(items, axis=0)

    @tf.function
    def gather(self, indices):
        """Gather nested structures.

        Args:
            indices: The indices to gather values.

        Returns:
            The nested structure.
        """
        tf.debugging.assert_greater(tf.shape(indices)[0], 0)
        items = tf.numpy_function(self._gaher_fn, (indices,), tf.dtypes.string)
        items.set_shape([None])

        decoded_items = tf.nest.map_structure(
            lambda x: x[None, ...],
            self.decode(items[0]))
        if tf.greater(many - 1, 0):
            for index in tf.range(num_items - 1):
                decoded_items = tf.nest.map_structure(
                    lambda x, y: tf.concat([x, y[None]], axis=0),
                    decoded_items,
                    self.decode(items[index]))
        return decoded_items

    def _len_fn(self, index):
        item = self._pipe.llen(self._key)
        if item:
            item = item[1]
        return int(item.decode())

    @tf.function
    def len(self):
        """Get length of the list.

        Returns:
            The length of the list.
        """
        length = tf.numpy_function(self._len_fn, (), tf.dtypes.int64)
        length = tf.ensure_shape(length, [])
        return length

    def _set_fn(self, index, value):
        self._pipe.lset(self._key, index.item(), value)

    @tf.function
    def set(self, index, structure):
        """Set a nested structure.

        Args:
            index: The index to set the value.
            structure: The nested structure.
        """
        buffer = self.encode(structure)
        tf.numpy_function(self._set_fn, (index, buffer), ())

    def _slice_get_fn(self, start, stop, step):
        start = start.item()
        stop = stop.item()
        step = step.item()

        length = None
        if start != abs(start):
            length = self._pipe.llen(self._key)
            start = length + start + 1
        if stop != abs(stop):
            if length is None:
                length = self._pipe.llen(self._key)
            stop = length + stop + 1

        pipeline = self._pipe.pipeline()
        pipeline.multi()
        for index in range(start, stop, step):
            value = pipeline.lindex(self._key, index)
        items = pipeline.execute()

        if items:
            stacked_items = np.stack(items, axis=0)
            return stacked_items, stacked_items.shape[0]
        return np.asarray([b'']), 0

    @tf.function
    def slice_get(self, start=None, stop=None, step=None):
        """Slice a nested structure.

        Args:
            start: The slice start.
            stop: The slice stop.
            step: The slice step.

        Returns:
            Nested structure of tensors. The first dimension of the tensor
                will be 0 if there are no elements in the list.
        """
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

        items, num_items = tf.numpy_function(
            self._slice_get_fn,
            (start, stop, step),
            (tf.dtypes.string, tf.dtypes.int64))
        num_items = tf.ensure_shape(num_items, [])

        if tf.equal(num_items, 0):
            return debugging_ops.mock_spec(tf.TensorShape([0]), self.specs)

        items.set_shape([None])
        decoded_items = tf.nest.map_structure(
            lambda x: x[None, ...],
            self.decode(items[0]))

        if tf.greater(num_items - 1, 0):
            for index in tf.range(num_items - 1):
                decoded_items = tf.nest.map_structure(
                    lambda x, y: tf.concat([x, y[None]], axis=0),
                    decoded_items,
                    self.decode(items[index]))
        return decoded_items


class Condition(object):

    def __init__(self, pipe, key):
        """Creates a new Condition.

        This creates a distributed condition datastructure.

        Args:
            pipe: The redis server.
            key: The redis key for the queue.
        """
        self._pipe = pipe
        self._key = key

    def _wait_fn(self, w_id):
        w_id_str = str(w_id.item())
        pipeline = self._pipe.pipeline()
        pipeline.multi()
        pipeline.rpush(self._key, w_id_str)
        pipeline.blpop(self._key + w_id_str)
        pipeline.execute()

    @tf.function
    def wait(self, w_id):
        """Block until a producer notifies this id.

        Args:
            w_id: The id to send to the producer.
        """
        with tf.control_dependencies([
                tf.numpy_function(self._wait_fn, (w_id,), ())]):
            return

    def _notify_fn(self, w_id_):
        self._pipe.rpush(self._key + str(w_id_.item()), 1)

    @tf.function
    def notify(self, w_id):
        """Notifies the id."""
        with tf.control_dependencies([
                tf.numpy_function(self._notify_fn, (w_id,), ())]):
            return

    def _notify_all_fn(self):
        pipeline = self._pipe.pipeline()
        pipeline.multi()
        pipeline.lrange(self._key, 0, -1)
        pipeline.delete(self._key)
        [w_ids, _] = pipeline.execute()
        pipeline = self._pipe.pipeline()
        pipeline.multi()
        for w_id in w_ids:
            pipeline.rpush(self._key + str(w_id.decode()), 1)
        _ = pipeline.execute()

    @tf.function
    def notify_all(self):
        """Notifies all active ids."""
        with tf.control_dependencies([
                tf.numpy_function(self._notify_all_fn, (), ())]):
            return


class Value(TensorCodec):

    def __init__(self, pipe, key, dtypes):
        """Creates a new Value.

        This creates a distributed value datastructure.

        Args:
            pipe: The redis server.
            key: The redis key for the value.
            dtypes: The possibly nested structure containing tensor
                dtypes.
        """
        super(Value, self).__init__(dtypes)
        self._pipe = pipe
        self._key = key

    def _set_fn(self, buffer):
        self._pipe.set(self._key, buffer)

    @tf.function
    def set(self, structure):
        """Set the nested structure.

        Args:
            structure: The nested structure.
        """
        buffer = self.encode(structure)
        with tf.control_dependencies([
                tf.numpy_function(self._set_fn, (buffer,), ())]):
            return

    def _get_fn(self):
        item = self._pipe.get(self._key)
        return item

    @tf.function
    def get(self):
        """Get the nested structure.

        Returns:
            The nested structure.
        """
        item = tf.numpy_function(self._get_fn, (), tf.dtypes.string)
        item = tf.ensure_shape(item, [])
        decoded_item = self.decode(item)
        return decoded_item


class Event(object):

    def __init__(self, pipe, num_index, key):
        """Creates a new Event.

        This creates a distributed Event datastructure.

        Args:
            pipe: The redis server.
            num_index: The number of indices.
            key: The redis key for the register.
        """
        self._pipe = pipe
        self._num_index = num_index
        self._key = key

    def _set_fn(self, w_id):
        self._pipe.set(self._key + str(w_id.item()), 1)

    @tf.function
    def set(self, w_id):
        """Set the event."""
        with tf.control_dependencies([
                tf.numpy_function(self._set_fn, (w_id,), ())]):
            return

    def _unset_fn(self, w_id):
        self._pipe.set(self._key + str(w_id.item()), 0)

    @tf.function
    def unset(self, w_id):
        """Unset the event."""
        with tf.control_dependencies([
                tf.numpy_function(self._unset_fn, (w_id,), ())]):
            return

    def _set_all_fn(self):
        pipeline = self._pipe.pipeline()
        pipeline.multi()
        for index in range(self._num_index):
            pipeline.set(self._key + str(index), 1)
        _ = pipeline.execute()

    @tf.function
    def set_all(self):
        """Set all events."""
        with tf.control_dependencies([
                tf.numpy_function(self._set_all_fn, (), ())]):
            return

    def _get_fn(self, w_id):
        return bool(int(self._pipe.get(self._key + str(w_id.item())).decode()))

    @tf.function
    def get(self, w_id):
        """Get the nested structure.

        Returns:
            The nested structure.
        """
        return tf.numpy_function(self._get_fn, (w_id,), tf.dtypes.bool)
