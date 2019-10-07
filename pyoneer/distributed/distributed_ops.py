from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import collections
import tensorflow as tf

from pyoneer.debugging import debugging_ops


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

        Implementation of a distributed queue data-structure. This implements
            a blocking queue similar to `multiprocessing.Queue`.

        For example:

            ```python
            dtypes = (tf.dtypes.float32, tf.dtypes.float32)
            structure = (tf.fill([10, 5, 10], tf.cast(-12, tf.dtypes.float32)),
                        tf.fill([10], tf.cast(100, tf.dtypes.float32)))

            # Create a queue data-structure.
            pipe = redis.Redis(host=host, port=port)
            queue = pynr.distributed.Queue(pipe, 'MyQueue', dtypes=dtypes)

            # Enqueue a structure.
            queue.enqueue(structure)

            # Block until the structure is enqueued.
            queue.dequeue(encoded_structure) == structure
            ```

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
        """Dequeue many nested structures.

        Args:
            many: The number of tensors to dequeue and stack together.

        Returns:
            The nested structure of tensors stacked along the first axis
                (Each element of the nested structure is now shape [many x ...]).
        """
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

        Implementation of a distributed list datastructure. The only blocking
            operation is done when `pop()` is called. The rest of the operations
            are not blocking, which makes this vulnerable to empty return values,
            which are not encoded at the moment, so be careful.

        For example:

            ```python
            structure = (tf.fill([10], tf.cast(100, tf.dtypes.float32)),)
            specs = tf.nest.map_structure(
                lambda t: tf.TensorSpec(t.shape.as_list(), t.dtype), structure)

            # Create a new list.
            pipe = redis.Redis(host=host, port=port)
            r = pynr.distributed.List(pipe, 'MyList', specs)

            # Append a structure.
            r.append(structure)

            # Pop the structure.
            r.pop() == structure
            ```

        Args:
            pipe: The redis server.
            key: The redis key for the queue. Each name is unique to
                the corresponding shared memory.
            specs: The possibly nested structure containing `tf.TensorSpec`s.
        """
        dtypes = tf.nest.map_structure(lambda s: s.dtype, specs)
        super(List, self).__init__(dtypes)
        self._specs = specs
        self._pipe = pipe
        self._key = key

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
        return self.len()

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
                    self.decode(items[index]))
        return decoded_items

    def _scatter_fn(self, indices, values):
        pipeline = self._pipe.pipeline()
        pipeline.multi()
        for index, value in zip(indices, np.unstack(values, axis=0)):
            pipeline.lset(self._key, index.item(), value)
        pipeline.execute()

    @tf.function
    def scatter(self, indices, values):
        """Scatter nested structures.

        Args:
            indices: The indices to scatter values.
            indices: Values to scatter.
        """
        encoded_values = tf.map_fn(self.encode, values, dtype=tf.dtypes.string)
        tf.numpy_function(self._scatter_fn, (indices, encoded_values))

    def _len_fn(self):
        item = self._pipe.llen(self._key)
        return item

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


class Condition(object):

    def __init__(self, pipe, key):
        """Creates a new Condition.

        Implementation of a distributed condition datastructure.
            Conditions are locking mechanisms controlled by another
            source other than the consumer.

        For example:

            ```python
            pipe = redis.Redis(host=host, port=port)
            condition = pynr.distributed.Condition(pipe, 'MyCondition')

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
            pipe: The redis server.
            key: The redis key for the condition.
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


class Lock(object):

    def __init__(self, pipe, key):
        """Creates a new Lock.

        Implementation of a distributed lock datastructure. This is
            a simple lock mechanism that uses redis-native blocking.

        For example:

            ```python
            pipe = redis.Redis(host=host, port=port)
            lock = pynr.distributed.Lock(pipe, 'MyLock')

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
            pipe: The redis server.
            key: The redis key for the condition.
        """
        self._pipe = pipe
        self._key = key
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

    def __init__(self, pipe, key, dtypes):
        """Creates a new Value.

        Implementation of a distributed value datastructure. This allows you to
            read/write an arbitrary object. Useful for a parameter server.

        For example:

            ```
            dtypes = (tf.dtypes.float32,)
            structure = (tf.fill([10], tf.cast(100, tf.dtypes.float32)),)

            pipe = redis.Redis(host=host, port=port)
            value = pynr.distributed.Value(pipe, 'MyValue', dtypes)

            # Set the value.
            value.set(structure)

            # Get the value.
            r.get() == structure
            ```

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

        Implementation of a distributed event datastructure.
            This is similar to a `threading.Event`, but a vector of events.
            You can `set`, `get` and `unset` this individually or set all
            events with `set_all`. It could be used as a simple flag for
            unblocking control flow, such as protect a parameter server to
            be accessed only when values have been updated.

        For example:

            ```python
            pipe = redis.Redis(host=host, port=port)
            event = distributed_ops.Event(pipe, 2, 'MyEvent')

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
        """Set the event by id.

        Args:
            w_id: The event id.
        """
        tf.numpy_function(self._set_fn, (w_id,), ())

    def _unset_fn(self, w_id):
        self._pipe.set(self._key + str(w_id.item()), 0)

    @tf.function
    def unset(self, w_id):
        """Unset the event by id.

        Args:
            w_id: The event id.
        """
        tf.numpy_function(self._unset_fn, (w_id,), ())

    def _set_all_fn(self):
        pipeline = self._pipe.pipeline()
        pipeline.multi()
        for index in range(self._num_index):
            pipeline.set(self._key + str(index), 1)
        _ = pipeline.execute()

    @tf.function
    def set_all(self):
        """Set all event ids."""
        tf.numpy_function(self._set_all_fn, (), ())

    def _get_fn(self, w_id):
        return bool(int(self._pipe.get(self._key + str(w_id.item())).decode()))

    @tf.function
    def get(self, w_id):
        """Get the value of the event by id.

        Returns:
            If the event is set.
        """
        return tf.numpy_function(self._get_fn, (w_id,), tf.dtypes.bool)
