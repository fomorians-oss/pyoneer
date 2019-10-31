from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import collections
import tensorflow as tf

from pyoneer.debugging import debugging_ops


_DEFAULT_PIPE = None

def set_default_pipe(pipe):
    """Set the default pipe for distributed communication."""
    global _DEFAULT_PIPE
    _DEFAULT_PIPE = pipe


def get_default_pipe():
    """Get the default pipe for distributed communication.

    Returns:
        The pipe associated with the current backend.
        For now, only `redis` is supported.
    """
    global _DEFAULT_PIPE
    return _DEFAULT_PIPE


def assign_key_name(name, default_name):
    if name is None:
        name = default_name
    with tf.name_scope(name) as base_name:
        return base_name


class TensorCodec(object):

    def __init__(self, dtypes):
        """Creates a new TensorCodec.

        This creates a codec that encodes structures to strings
            and decodes messages back to the structures.

        For example:

            ```python
            dtypes = (tf.dtypes.float32, tf.dtypes.float32)
            structure = (tf.fill([10, 5, 10], tf.cast(-12, tf.dtypes.float32)),
                        tf.fill([10], tf.cast(100, tf.dtypes.float32)))

            codec = pynr.distributed.TensorCodec(dtypes=dtypes)
            encoded_structure = codec.encode(structure)  # A sequence of bytes.

            codec.decode(encoded_structure) == structure
            ```

        Args:
            dtypes: The possibly nested structure containing tensor
                dtypes for all expected encoded structures.
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


class Deque(TensorCodec):

    def __init__(self, dtypes, capacity=None, name=None, pipe=None):
        """Creates a new Deque.

        Implementation of a distributed queue/stack data-structure. This implements
            a blocking queue similar to `multiprocessing.Queue` and `deque`.

        For example:

            ```python
            dtypes = (tf.dtypes.float32, tf.dtypes.float32)
            structure = (tf.fill([10, 5, 10], tf.cast(-12, tf.dtypes.float32)),
                        tf.fill([10], tf.cast(100, tf.dtypes.float32)))

            # Create a queue data-structure with a deque.
            deque = pynr.distributed.Deque(dtypes)

            # Enqueue a structure.
            deque.append(structure)

            # Block until the structure is enqueued.
            deque.popleft(encoded_structure) == structure
            ```

        Args:
            dtypes: The possibly nested structure containing tensor
                dtypes.
            capacity: (optional) The max size of the deque. When the size is reached, a
                corresponding number of elements will be removed from the deque in the
                opposite end that they are added.
            name: (optional) The redis key, corresponds to the `tf.namescope` that this
                name is added to.
            pipe: (optional) The redis server. If not provided, this defaults to the
                result returned by `pynr.distributed.get_default_pipe()`.
        """
        super(Deque, self).__init__(dtypes)
        if pipe is None:
            pipe = get_default_pipe()
            assert pipe is not None, ('No default pipe set, must use `set_default_pipe`'
                                      'or pass a pipe that is not `None`.')
        if capacity is None:
            capacity = -1
        self._capacity = tf.convert_to_tensor(capacity, tf.dtypes.int64)
        self._pipe = pipe
        self._key = assign_key_name(name, 'Deque')

    @property
    def capacity(self):
        return self._capacity

    def _append_fn(self, buffer, capacity):
        pipeline = self._pipe.pipeline()
        pipeline.multi()
        pipeline.rpush(self._key, buffer)
        if capacity != -1:
            pipeline.ltrim(self._key, 1, capacity + 1)
        pipeline.execute()

    @tf.function
    def append(self, structure):
        """Append a nested structure to the right.

        Args:
            structure: The nested structure.
        """
        buffer = self.encode(structure)
        tf.numpy_function(self._append_fn, (buffer, self._capacity), ())

    def _append_left_fn(self, buffer, capacity):
        pipeline = self._pipe.pipeline()
        pipeline.multi()
        pipeline.lpush(self._key, buffer)
        if capacity != -1:
            pipeline.ltrim(self._key, 0, capacity)
        pipeline.execute()

    @tf.function
    def append_left(self, structure):
        """Append a nested structure to the left.

        Args:
            structure: The nested structure.
        """
        buffer = self.encode(structure)
        tf.numpy_function(self._append_left_fn, (buffer, self._capacity), ())

    def _pop_fn(self):
        item = self._pipe.brpop(self._key)
        if item:
            item = item[1]
        return item

    @tf.function
    def pop(self):
        """Pop a nested structure from the right.

        Returns:
            The nested structure.
        """
        item = tf.numpy_function(self._pop_fn, (), tf.dtypes.string)
        item = tf.ensure_shape(item, [])
        decoded_item = self.decode(item)
        return decoded_item

    def _popleft_fn(self):
        item = self._pipe.blpop(self._key)
        if item:
            item = item[1]
        return item

    @tf.function
    def popleft(self):
        """Pop a nested structure from the left.

        Returns:
            The nested structure.
        """
        item = tf.numpy_function(self._popleft_fn, (), tf.dtypes.string)
        item = tf.ensure_shape(item, [])
        decoded_item = self.decode(item)
        return decoded_item


class Condition(object):

    def __init__(self, name=None, pipe=None):
        """Creates a new Condition.

        Implementation of a distributed condition datastructure.
            Conditions are locking mechanisms controlled by another
            source other than the consumer.

        For example:

            ```python
            condition = pynr.distributed.Condition()

            def consumer_fn(cond, w_id):
                cond.wait(w_id)
                print('Hello from {}!'.format(w_id))

            consumers = []
            for consumer_id in range(4):
                consumer = threading.Thread(target=consumer_fn,
                                            args=(condition, consumer_id))
                consumer.start()
                time.sleep(1)
                consumers.append(consumer)

            # Tell the consumers to stop waiting.
            condition.notify_all()

            # Cleanup.
            for consumer in consumers:
                consumer.join()
            ```

        Args:
            name: (optional) The redis key, corresponds to the `tf.namescope` that this
                name is added to.
            pipe: (optional) The redis server. If not provided, this defaults to the
                result returned by `pynr.distributed.get_default_pipe()`.
        """
        if pipe is None:
            pipe = get_default_pipe()
            assert pipe is not None, ('No default pipe set, must use `set_default_pipe`'
                                      'or pass a pipe that is not `None`.')
        self._pipe = pipe
        self._key = assign_key_name(name, 'Condition')

    def _wait_fn(self, w_id):
        w_id_str = str(w_id.item())
        self._pipe.rpush(self._key, w_id_str)
        self._pipe.blpop(self._key + w_id_str)

    @tf.function
    def wait(self, w_id):
        """Block until a producer notifies this id.

        Args:
            w_id: The id to send to the producer.
        """
        tf.numpy_function(self._wait_fn, (w_id,), ())

    def _notify_fn(self, w_id_):
        self._pipe.rpush(self._key + str(w_id_.item()), 1)

    @tf.function
    def notify(self, w_id):
        """Notifies the consumer by id.

        Args:
            w_id: The id to send to the consumer.
        """
        tf.numpy_function(self._notify_fn, (w_id,), ())

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
        """Notifies all consumer ids sent to the producer."""
        tf.numpy_function(self._notify_all_fn, (), ())


class Counter(object):

    def __init__(self, dtype=tf.dtypes.int64, name=None, pipe=None):
        """Creates a new Counter.

        This implements an incremental counter (-increment = decrement).

        Args:
            dtype: The dtype of the counter variable returned.
            name: (optional) The redis key, corresponds to the `tf.namescope` that this
                name is added to.
            pipe: (optional) The redis server. If not provided, this defaults to the
                result returned by `pynr.distributed.get_default_pipe()`.
        """
        if pipe is None:
            pipe = get_default_pipe()
            assert pipe is not None, ('No default pipe set, must use `set_default_pipe`'
                                      'or pass a pipe that is not `None`.')
        self._dtype = dtype
        self._pipe = pipe
        self._key = assign_key_name(name, 'Counter')
        self._initialize()

    @property
    def dtype(self):
        return self._dtype

    def _initialize_fn(self):
        if not self._pipe.exists(self._key):
            self._pipe.set(self._key, 0)

    def _initialize(self):
        tf.numpy_function(self._initialize, (), ())

    def _increment_fn(self, offset):
        value = self._pipe.incrby(self._key, offset.item())
        return value

    @tf.function
    def increment(self, offset=1):
        """Increment the counter and return the new value.

        Returns:
            Tensor of type `dtype` corresponding to the currrent value.
        """
        offset = tf.cast(offset, tf.dtypes.int64)
        return tf.cast(
            tf.numpy_function(self._increment_fn, (offset,), tf.dtypes.int64),
            self._dtype)

    def _get_fn(self):
        return int(self._pipe.get(self._key).decode())

    @tf.function
    def get(self):
        """Get the value of the counter.

        Returns:
            Tensor of type `dtype` corresponding to the currrent value.
        """
        return tf.cast(
            tf.numpy_function(self._get_fn, (), tf.dtypes.int64),
            self._dtype)


class Lock(object):

    def __init__(self, name=None, pipe=None):
        """Creates a new Lock.

        Implementation of a distributed lock datastructure. This is
            a simple lock mechanism that uses redis-native blocking.

        For example:

            ```python
            lock = pynr.distributed.Lock()

            def consumer_fn(lock, w_id):
                with lock:
                    print('Hello from {}!'.format(w_id))
                    time.sleep(.5)

            consumers = []
            for consumer_id in range(4):
                consumer = threading.Thread(target=consumer_fn,
                                            args=(lock, consumer_id))
                consumer.start()
                consumers.append(consumer)

            # Cleanup.
            for consumer in consumers:
                consumer.join()
            ```

        Args:
            name: (optional) The redis key, corresponds to the `tf.namescope` that this
                name is added to.
            pipe: (optional) The redis server. If not provided, this defaults to the
                result returned by `pynr.distributed.get_default_pipe()`.
        """
        if pipe is None:
            pipe = get_default_pipe()
            assert pipe is not None, ('No default pipe set, must use `set_default_pipe`'
                                      'or pass a pipe that is not `None`.')
        self._pipe = pipe
        self._key = assign_key_name(name, 'Lock')
        self._initialize_fn()

    def _initialize_fn(self):
        if not self._pipe.exists(self._key):
            self._release_fn()

    def _acquire_fn(self):
        _ = self._pipe.brpop(self._key)

    @tf.function
    def acquire(self):
        """Acquire the lock."""
        tf.numpy_function(self._acquire_fn, (), ())

    def _release_fn(self):
        pipeline = self._pipe.pipeline()
        pipeline.multi()
        pipeline.rpush(self._key, 1)
        pipeline.ltrim(self._key, 0, 1)
        pipeline.execute()

    @tf.function
    def release(self):
        """Release the lock."""
        tf.numpy_function(self._release_fn, (), ())

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, type, value, traceback):
        self.release()


class Value(TensorCodec):

    def __init__(self, dtypes, name=None, pipe=None):
        """Creates a new Value.

        Implementation of a distributed value datastructure. This allows you to
            read/write an arbitrary object. Useful for a parameter server.

        For example:

            ```
            dtypes = (tf.dtypes.float32,)
            structure = (tf.fill([10], tf.cast(100, tf.dtypes.float32)),)

            value = pynr.distributed.Value()

            # Set the value.
            value.set(structure)

            # Get the value.
            r.get() == structure
            ```

        Args:
            dtypes: The possibly nested structure containing tensor
                dtypes.
            name: (optional) The redis key, corresponds to the `tf.namescope` that this
                name is added to.
            pipe: (optional) The redis server. If not provided, this defaults to the
                result returned by `pynr.distributed.get_default_pipe()`.
        """
        super(Value, self).__init__(dtypes)
        if pipe is None:
            pipe = get_default_pipe()
            assert pipe is not None, ('No default pipe set, must use `set_default_pipe`'
                                      'or pass a pipe that is not `None`.')
        self._pipe = pipe
        self._key = assign_key_name(name, 'Value')

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

    def __init__(self, name=None, pipe=None):
        """Creates a new Event.

        Implementation of a distributed event datastructure.
            This is similar to a `threading.Event`, but a vector of events.
            You can `set`, `get` and `unset` this individually or set all
            events with `set_all`. It could be used as a simple flag for
            unblocking control flow, such as protect a parameter server to
            be accessed only when values have been updated.

        For example:

            ```python
            event = distributed_ops.Event()

            # Set one event, unset the other.
            event.set(0)
            event.unset(1)

            event.get(0) == True
            event.get(1) == False

            # Set all the events.
            event.set_all()

            event.get(0) == True
            event.get(1) == True
            ```

        Args:
            name: (optional) The redis key, corresponds to the `tf.namescope` that this
                name is added to.
            pipe: (optional) The redis server. If not provided, this defaults to the
                result returned by `pynr.distributed.get_default_pipe()`.
        """
        if pipe is None:
            pipe = get_default_pipe()
            assert pipe is not None, ('No default pipe set, must use `set_default_pipe`'
                                      'or pass a pipe that is not `None`.')
        self._pipe = pipe
        self._key = assign_key_name(name, 'Event')

    def _set_fn(self, w_id):
        self._pipe.hset(self._key, str(w_id.item()), 1)

    @tf.function
    def set(self, w_id):
        """Set the event by id.

        Args:
            w_id: The event id.
        """
        tf.numpy_function(self._set_fn, (w_id,), ())

    def _unset_fn(self, w_id):
        self._pipe.hset(self._key, str(w_id.item()), 0)

    @tf.function
    def unset(self, w_id):
        """Unset the event by id.

        Args:
            w_id: The event id.
        """
        tf.numpy_function(self._unset_fn, (w_id,), ())

    def _set_all_fn(self):
        w_ids = self._pipe.hkeys(self._key)
        pipeline = self._pipe.pipeline()
        pipeline.multi()
        for w_id in w_ids:
            pipeline.hset(self._key, w_id, 1)
        _ = pipeline.execute()

    @tf.function
    def set_all(self):
        """Set all event ids."""
        tf.numpy_function(self._set_all_fn, (), ())

    def _get_fn(self, w_id):
        key = str(w_id.item())
        if not self._pipe.hexists(self._key, key):
            self._pipe.hset(self._key, key, 1)
            return True
        return bool(int(self._pipe.hget(self._key, key).decode()))

    @tf.function
    def get(self, w_id):
        """Get the value of the event by id.

        Returns:
            If the event is set.
        """
        return tf.numpy_function(self._get_fn, (w_id,), tf.dtypes.bool)
