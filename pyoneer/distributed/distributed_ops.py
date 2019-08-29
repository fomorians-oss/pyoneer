from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


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
        self._zero_offset = tf.cast(
            tf.strings.length(
                self._encode(tf.zeros([self._count], self._length_dtype))),
            self._length_dtype)
        self._offset = self._zero_offset

    @property
    def dtypes(self):
        return self._dtypes

    def _encode(self, tensor):
        return tf.io.serialize_tensor(tensor)

    def _decode(self, msg, dtype):
        return tf.io.parse_tensor(msg, dtype)

    def encode(self, item):
        """Encode a structure into a packed message.

        Args:
            item: A structure with the same structure as dtypes.

        Returns:
            The encoded message.
        """
        # Ensure the same structure.
        tf.nest.assert_same_structure(item, self.dtypes)

        lengths = []
        structure_msg = []

        def encode_w_lengths(tensor):
            msg = self._encode(tensor)
            structure_msg.append(msg)
            lengths.append(tf.cast(tf.strings.length(msg), self._length_dtype))
            return msg

        _ = tf.nest.map_structure(encode_w_lengths, item)
        lengths_tensor = tf.stack(lengths, axis=0)
        lengths_msg = self._encode(lengths_tensor)
        buffer = tf.strings.join([lengths_msg, tf.strings.join(structure_msg)])
        return buffer

    def decode(self, item):
        """Decode a packed message that represents a structure.

        Args:
            item: A string representing the encoded message.

        Returns:
            The decoded structure.
        """
        offset = self._offset

        # Get the lengths of the items in the encoded string.
        lengths_msg = tf.strings.substr(item, 0, self._offset)
        lengths = self._decode(lengths_msg, self._length_dtype)
        lengths.set_shape([self._count])

        def decode_w_lengths(length, dtype):
            item_structure = tf.strings.substr(item, self._offset, length)
            self._offset = self._offset + length
            return self._decode(item_structure, dtype)

        # Decode the item into the same encoded structure.
        packed_lengths = tf.nest.pack_sequence_as(
            self._dtypes, tf.unstack(lengths, axis=0))
        structure = tf.nest.map_structure(
            decode_w_lengths, packed_lengths, self._dtypes)

        self._offset = offset
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

    def enqueue(self, structure):
        """Enqueue a nested structure.

        Args:
            structure: The nested structure.
        """
        buffer = self.encode(structure)

        def enqueue_fn(buffer):
            self._pipe.rpush(self._key, buffer.numpy())

        tf.py_function(enqueue_fn, (buffer,), ())

    def dequeue(self):
        """Dequeue a nested structure.

        Returns:
            The nested structure.
        """
        def dequeue_fn():
            item = self._pipe.blpop(self._key)
            if item:
                item = item[1]
            return tf.nest.flatten([self.decode(item)])

        flat_tensors = tf.py_function(dequeue_fn, (),
                                      tf.nest.flatten([self.dtypes]))
        tensors = tf.nest.pack_sequence_as(self.dtypes, flat_tensors)
        return tensors

    def dequeue_many(self, many, axis=0):
        """Dequeue many nested structures and stack them.

        Args:
            many: The number of nested structures to dequeue.
            axis: The axis to stack along.

        Returns:
            The nested, stacked structure.
        """
        def dequeue_fn():
            decoded_items = []
            for _ in range(many):
                item = self._pipe.blpop(self._key)
                if item:
                    item = item[1]
                decoded = tf.nest.flatten([self.decode(item)])
                decoded_items.append(decoded)
            return tf.nest.map_structure(lambda *x: tf.stack(x, axis=axis),
                                         *decoded_items)

        flat_tensors = tf.py_function(dequeue_fn, (),
                                      tf.nest.flatten([self.dtypes]))
        tensors = tf.nest.pack_sequence_as(self.dtypes, flat_tensors)
        return tensors


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

    def wait(self, w_id):
        """Block until a producer notifies this id.

        Args:
            w_id: The id to send to the producer.
        """
        def wait_fn():
            self._pipe.rpush(self._key, str(w_id))
            _ = self._pipe.blpop(self._key + str(w_id))

        tf.py_function(wait_fn, (), ())

    def notify_first(self):
        """Notifies the first id."""
        def notify_first_fn():
            ids = []
            w_id = self._pipe.lpop(self._key)
            if w_id:
                self._pipe.rpush(self._key + str(w_id.decode()), 1)

        tf.py_function(notify_first_fn, (), ())

    def notify_all(self):
        """Notifies all active ids."""
        def notify_all_fn():
            w_ids = []
            while True:
                w_id = self._pipe.lpop(self._key)
                if not w_id:
                    break
                w_ids.append(w_id)

            for w_id in w_ids:
                self._pipe.rpush(self._key + str(w_id.decode()), 1)

        tf.py_function(notify_all_fn, (), ())


class Register(TensorCodec):

    def __init__(self, pipe, key, dtypes):
        """Creates a new Register.

        This creates a distributed register datastructure.

        Args:
            pipe: The redis server.
            key: The redis key for the register.
            dtypes: The possibly nested structure containing tensor
                dtypes.
        """
        super(Register, self).__init__(dtypes)
        self._pipe = pipe
        self._key = key

    def set(self, structure):
        """Set the nested structure.

        Args:
            structure: The nested structure.
        """
        buffer = self.encode(structure)

        def set_fn(buffer):
            self._pipe.set(self._key, buffer.numpy())

        tf.py_function(set_fn, (buffer,), ())

    def get(self):
        """Get the nested structure.

        Returns:
            The nested structure.
        """
        def get_fn():
            item = self._pipe.get(self._key)
            return tf.nest.flatten([self.decode(item)])

        flat_tensors = tf.py_function(get_fn, (),
                                      tf.nest.flatten([self.dtypes]))
        tensors = tf.nest.pack_sequence_as(self.dtypes, flat_tensors)
        return tensors


class MultiEvent(object):

    def __init__(self, pipe, index, num_index, key):
        """Creates a new MultiEvent.

        This creates a distributed register datastructure.

        Args:
            pipe: The redis server.
            index: The corresponding index.
            num_index: The number of indices.
            key: The redis key for the register.
        """
        self._pipe = pipe
        self._index = index
        self._num_index = num_index
        self._key = key

    def set(self):
        """Set the event."""
        def set_fn():
            self._pipe.set(self._key + str(self._index), 1)

        tf.py_function(set_fn, (), ())

    def unset(self):
        """Unset the event."""
        def unset_fn():
            self._pipe.set(self._key + str(self._index), 0)

        tf.py_function(unset_fn, (), ())

    def set_all(self):
        """Set all events."""
        def set_fn():
            for index in range(self._num_index):
                self._pipe.set(self._key + str(index), 1)

        tf.py_function(set_fn, (), ())

    def get(self):
        """Get the nested structure.

        Returns:
            The nested structure.
        """
        def get_fn():
            return bool(self._pipe.get(self._key + str(self._index)).decode())

        return tf.py_function(get_fn, (), tf.dtypes.bool)
