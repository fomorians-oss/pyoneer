from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import threading

try:
    import redis
except ImportError as e:
    print(
        (
            "Redis must be installed to use `pynr.distributed`. "
            "Run `pip install redis`."
        )
    )
    print("To run these tests, start `redis-server`.")
    exit()

import tensorflow as tf

from pyoneer.distributed import distributed_ops


class TensorCodecTest(tf.test.TestCase):
    def testTuple(self):
        # 1-tuple.
        dtypes = (tf.dtypes.float32,)
        structure = (tf.fill([10], tf.cast(100, tf.dtypes.float32)),)

        codec = distributed_ops.TensorCodec(dtypes=dtypes)
        encoded_msg = codec.encode(structure)
        decoded = codec.decode(encoded_msg)
        tf.nest.map_structure(self.assertAllEqual, decoded, structure)

        # 2-tuple.
        dtypes = (tf.dtypes.float32, tf.dtypes.float32)
        structure = (
            tf.fill([10, 5, 10], tf.cast(-12, tf.dtypes.float32)),
            tf.fill([10], tf.cast(100, tf.dtypes.float32)),
        )

        codec = distributed_ops.TensorCodec(dtypes=dtypes)
        encoded_msg = codec.encode(structure)
        decoded = codec.decode(encoded_msg)

        tf.nest.map_structure(self.assertAllEqual, decoded, structure)

        # nested 2-tuple.
        dtypes = (
            (tf.dtypes.float32, tf.dtypes.float32),
            (tf.dtypes.float32, tf.dtypes.float32),
        )
        structure = (
            (
                tf.fill([10, 5, 10], tf.cast(-12, tf.dtypes.float32)),
                tf.fill([10], tf.cast(100, tf.dtypes.float32)),
            ),
            (
                tf.fill([10, 5, 10], tf.cast(-12, tf.dtypes.float32)),
                tf.fill([10], tf.cast(100, tf.dtypes.float32)),
            ),
        )

        codec = distributed_ops.TensorCodec(dtypes=dtypes)
        encoded_msg = codec.encode(structure)
        decoded = codec.decode(encoded_msg)

        tf.nest.map_structure(self.assertAllEqual, decoded, structure)

    def testList(self):
        # 1-list.
        dtypes = [tf.dtypes.float32]
        structure = [tf.fill([10], tf.cast(100, tf.dtypes.float32))]

        codec = distributed_ops.TensorCodec(dtypes=dtypes)
        encoded_msg = codec.encode(structure)
        decoded = codec.decode(encoded_msg)

        tf.nest.map_structure(self.assertAllEqual, decoded, structure)

        # 2-list.
        dtypes = [tf.dtypes.float32, tf.dtypes.float32]
        structure = [
            tf.fill([10, 5, 10], tf.cast(-12, tf.dtypes.float32)),
            tf.fill([10], tf.cast(100, tf.dtypes.float32)),
        ]

        codec = distributed_ops.TensorCodec(dtypes=dtypes)
        encoded_msg = codec.encode(structure)
        decoded = codec.decode(encoded_msg)

        tf.nest.map_structure(self.assertAllEqual, decoded, structure)

        # nested 2-list.
        dtypes = [
            [tf.dtypes.float32, tf.dtypes.float32],
            [tf.dtypes.float32, tf.dtypes.float32],
        ]
        structure = [
            [
                tf.fill([10, 5, 10], tf.cast(-12, tf.dtypes.float32)),
                tf.fill([10], tf.cast(100, tf.dtypes.float32)),
            ],
            [
                tf.fill([10, 5, 10], tf.cast(-12, tf.dtypes.float32)),
                tf.fill([10], tf.cast(100, tf.dtypes.float32)),
            ],
        ]

        codec = distributed_ops.TensorCodec(dtypes=dtypes)
        encoded_msg = codec.encode(structure)
        decoded = codec.decode(encoded_msg)

        tf.nest.map_structure(self.assertAllEqual, decoded, structure)

    def testDict(self):
        # 1-dict.
        dtypes = {"a": tf.dtypes.float32}
        structure = {"a": tf.fill([10, 5, 10], tf.cast(-12, tf.dtypes.float32))}

        codec = distributed_ops.TensorCodec(dtypes=dtypes)
        encoded_msg = codec.encode(structure)
        decoded = codec.decode(encoded_msg)

        tf.nest.map_structure(self.assertAllEqual, decoded, structure)

        # 2-dict.
        dtypes = {"a": tf.dtypes.float32, "b": tf.dtypes.float32}
        structure = {
            "a": tf.fill([10, 5, 10], tf.cast(-12, tf.dtypes.float32)),
            "b": tf.fill([10], tf.cast(100, tf.dtypes.float32)),
        }

        codec = distributed_ops.TensorCodec(dtypes=dtypes)
        encoded_msg = codec.encode(structure)
        decoded = codec.decode(encoded_msg)

        tf.nest.map_structure(self.assertAllEqual, decoded, structure)

        # nested 2-dict.
        dtypes = {
            "a": {"a": tf.dtypes.float32, "b": tf.dtypes.float32},
            "b": {"a": tf.dtypes.float32, "b": tf.dtypes.float32},
        }
        structure = {
            "a": {
                "a": tf.fill([10, 5, 10], tf.cast(-12, tf.dtypes.float32)),
                "b": tf.fill([10], tf.cast(100, tf.dtypes.float32)),
            },
            "b": {
                "a": tf.fill([10, 5, 10], tf.cast(-12, tf.dtypes.float32)),
                "b": tf.fill([10], tf.cast(100, tf.dtypes.float32)),
            },
        }

        codec = distributed_ops.TensorCodec(dtypes=dtypes)
        encoded_msg = codec.encode(structure)
        decoded = codec.decode(encoded_msg)

        tf.nest.map_structure(self.assertAllEqual, decoded, structure)


class DequeTest(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        distributed_ops.set_default_pipe(redis.Redis(host="127.0.0.1", port=6379, db=0))

    def testEnqueueDequeueTuple(self):
        # 1-tuple.
        dtypes = (tf.dtypes.float32,)
        structure = (tf.fill([10], tf.cast(100, tf.dtypes.float32)),)

        with tf.name_scope("Tuple"):
            q = distributed_ops.Deque(dtypes)

        q.append(structure)
        actual_structure = q.popleft()

        tf.nest.map_structure(self.assertAllEqual, structure, actual_structure)

        # 2-tuple.
        dtypes = (tf.dtypes.float32, tf.dtypes.float32)
        structure = (
            tf.fill([10, 5, 10], tf.cast(-12, tf.dtypes.float32)),
            tf.fill([10], tf.cast(100, tf.dtypes.float32)),
        )

        with tf.name_scope("TwoTuple"):
            q = distributed_ops.Deque(dtypes)

        q.append(structure)
        actual_structure = q.popleft()
        tf.nest.map_structure(self.assertAllEqual, structure, actual_structure)

        # nested tuple.
        dtypes = (
            (tf.dtypes.float32, tf.dtypes.float32),
            (tf.dtypes.float32, tf.dtypes.float32),
        )
        structure = (
            (
                tf.fill([10, 5, 10], tf.cast(-12, tf.dtypes.float32)),
                tf.fill([10], tf.cast(100, tf.dtypes.float32)),
            ),
            (
                tf.fill([10, 5, 10], tf.cast(-12, tf.dtypes.float32)),
                tf.fill([10], tf.cast(100, tf.dtypes.float32)),
            ),
        )

        with tf.name_scope("NestedTuple"):
            q = distributed_ops.Deque(dtypes)

        q.append(structure)
        actual_structure = q.popleft()
        tf.nest.map_structure(self.assertAllEqual, structure, actual_structure)

    def testEnqueueDequeueList(self):
        # 1-list.
        dtypes = [tf.dtypes.float32]
        structure = [tf.fill([10], tf.cast(100, tf.dtypes.float32))]

        with tf.name_scope("List"):
            q = distributed_ops.Deque(dtypes)

        q.append(structure)
        actual_structure = q.popleft()
        tf.nest.map_structure(self.assertAllEqual, structure, actual_structure)

        # 2-list.
        dtypes = [tf.dtypes.float32, tf.dtypes.float32]
        structure = [
            tf.fill([10, 5, 10], tf.cast(-12, tf.dtypes.float32)),
            tf.fill([10], tf.cast(100, tf.dtypes.float32)),
        ]

        with tf.name_scope("TwoList"):
            q = distributed_ops.Deque(dtypes)

        q.append(structure)
        actual_structure = q.popleft()
        tf.nest.map_structure(self.assertAllEqual, structure, actual_structure)

        # nested 2-list.
        dtypes = [
            [tf.dtypes.float32, tf.dtypes.float32],
            [tf.dtypes.float32, tf.dtypes.float32],
        ]
        structure = [
            [
                tf.fill([10, 5, 10], tf.cast(-12, tf.dtypes.float32)),
                tf.fill([10], tf.cast(100, tf.dtypes.float32)),
            ],
            [
                tf.fill([10, 5, 10], tf.cast(-12, tf.dtypes.float32)),
                tf.fill([10], tf.cast(100, tf.dtypes.float32)),
            ],
        ]

        with tf.name_scope("NestedList"):
            q = distributed_ops.Deque(dtypes)

        q.append(structure)
        actual_structure = q.popleft()
        tf.nest.map_structure(self.assertAllEqual, structure, actual_structure)

    def testEnqueueDequeueDict(self):
        # 1-dict.
        dtypes = {"a": tf.dtypes.float32}
        structure = {"a": tf.fill([10, 5, 10], tf.cast(-12, tf.dtypes.float32))}

        with tf.name_scope("Dict"):
            q = distributed_ops.Deque(dtypes)

        q.append(structure)
        actual_structure = q.popleft()
        tf.nest.map_structure(self.assertAllEqual, structure, actual_structure)

        # 2-dict.
        dtypes = {"a": tf.dtypes.float32, "b": tf.dtypes.float32}
        structure = {
            "a": tf.fill([10, 5, 10], tf.cast(-12, tf.dtypes.float32)),
            "b": tf.fill([10], tf.cast(100, tf.dtypes.float32)),
        }

        with tf.name_scope("TwoDict"):
            q = distributed_ops.Deque(dtypes)

        q.append(structure)
        actual_structure = q.popleft()
        tf.nest.map_structure(self.assertAllEqual, structure, actual_structure)

        # nested 2-dict.
        dtypes = {
            "a": {"a": tf.dtypes.float32, "b": tf.dtypes.float32},
            "b": {"a": tf.dtypes.float32, "b": tf.dtypes.float32},
        }
        structure = {
            "a": {
                "a": tf.fill([10, 5, 10], tf.cast(-12, tf.dtypes.float32)),
                "b": tf.fill([10], tf.cast(100, tf.dtypes.float32)),
            },
            "b": {
                "a": tf.fill([10, 5, 10], tf.cast(-12, tf.dtypes.float32)),
                "b": tf.fill([10], tf.cast(100, tf.dtypes.float32)),
            },
        }

        with tf.name_scope("NestedDict"):
            q = distributed_ops.Deque(dtypes)

        q.append(structure)
        actual_structure = q.popleft()
        tf.nest.map_structure(self.assertAllEqual, structure, actual_structure)


class ConditionTest(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        distributed_ops.set_default_pipe(redis.Redis(host="127.0.0.1", port=6379, db=0))

    def testWaitNotifyAll(self):
        num_consumers = 2
        condition = distributed_ops.Condition()

        def consumer_fn(cond, w_id):
            cond.wait(w_id)

        consumers = []
        for consumer_id in range(num_consumers):
            consumer = threading.Thread(
                target=consumer_fn, args=(condition, consumer_id)
            )
            consumer.start()
            time.sleep(3)
            consumers.append(consumer)

        condition.notify_all()
        for consumer in consumers:
            consumer.join()


class CounterTest(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        distributed_ops.set_default_pipe(redis.Redis(host="127.0.0.1", port=6379, db=0))

    def testIncrementAndGet(self):
        num_consumers = 3
        counter = distributed_ops.Counter()

        def consumer_fn(counter, w_id):
            print("Hello from {}!".format(w_id))
            counter.increment(1)

        consumers = []
        for consumer_id in range(num_consumers):
            consumer = threading.Thread(target=consumer_fn, args=(counter, consumer_id))
            consumer.start()
            consumers.append(consumer)

        for consumer in consumers:
            consumer.join()

        value = counter.get()
        self.assertAllEqual(value, num_consumers)


class LockTest(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        distributed_ops.set_default_pipe(redis.Redis(host="127.0.0.1", port=6379, db=0))

    def testAcquireRelease(self):
        num_consumers = 2
        lock = distributed_ops.Lock()

        def consumer_fn(lock, w_id):
            with lock:
                print("Hello from {}!".format(w_id))
                time.sleep(0.5)

        consumers = []
        for consumer_id in range(num_consumers):
            consumer = threading.Thread(target=consumer_fn, args=(lock, consumer_id))
            consumer.start()
            consumers.append(consumer)

        for consumer in consumers:
            consumer.join()


class EventTest(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        distributed_ops.set_default_pipe(redis.Redis(host="127.0.0.1", port=6379, db=0))

    def testSetGet(self):
        event = distributed_ops.Event()

        event.set(0)
        event.unset(1)
        self.assertAllEqual(event.get(0), True)
        self.assertAllEqual(event.get(1), False)

        event.set_all()
        self.assertAllEqual(event.get(0), True)
        self.assertAllEqual(event.get(1), True)


class ValueTest(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        distributed_ops.set_default_pipe(redis.Redis(host="127.0.0.1", port=6379, db=0))

    def testSetGetTuple(self):
        # 1-tuple.
        dtypes = (tf.dtypes.float32,)
        structure = (tf.fill([10], tf.cast(100, tf.dtypes.float32)),)

        with tf.name_scope("Tuple"):
            r = distributed_ops.Value(dtypes)

        r.set(structure)
        actual_structure = r.get()
        tf.nest.map_structure(self.assertAllEqual, structure, actual_structure)

        # 2-tuple.
        dtypes = (tf.dtypes.float32, tf.dtypes.float32)
        structure = (
            tf.fill([10, 5, 10], tf.cast(-12, tf.dtypes.float32)),
            tf.fill([10], tf.cast(100, tf.dtypes.float32)),
        )

        with tf.name_scope("TwoTuple"):
            r = distributed_ops.Value(dtypes)

        r.set(structure)
        actual_structure = r.get()
        tf.nest.map_structure(self.assertAllEqual, structure, actual_structure)

        # nested tuple.
        dtypes = (
            (tf.dtypes.float32, tf.dtypes.float32),
            (tf.dtypes.float32, tf.dtypes.float32),
        )
        structure = (
            (
                tf.fill([10, 5, 10], tf.cast(-12, tf.dtypes.float32)),
                tf.fill([10], tf.cast(100, tf.dtypes.float32)),
            ),
            (
                tf.fill([10, 5, 10], tf.cast(-12, tf.dtypes.float32)),
                tf.fill([10], tf.cast(100, tf.dtypes.float32)),
            ),
        )

        with tf.name_scope("NestedTuple"):
            r = distributed_ops.Value(dtypes)

        r.set(structure)
        actual_structure = r.get()
        tf.nest.map_structure(self.assertAllEqual, structure, actual_structure)

    def testSetGetList(self):
        # 1-list.
        dtypes = [tf.dtypes.float32]
        structure = [tf.fill([10], tf.cast(100, tf.dtypes.float32))]

        with tf.name_scope("List"):
            r = distributed_ops.Value(dtypes)

        r.set(structure)
        actual_structure = r.get()
        tf.nest.map_structure(self.assertAllEqual, structure, actual_structure)

        # 2-list.
        dtypes = [tf.dtypes.float32, tf.dtypes.float32]
        structure = [
            tf.fill([10, 5, 10], tf.cast(-12, tf.dtypes.float32)),
            tf.fill([10], tf.cast(100, tf.dtypes.float32)),
        ]

        with tf.name_scope("TwoList"):
            r = distributed_ops.Value(dtypes)

        r.set(structure)
        actual_structure = r.get()
        tf.nest.map_structure(self.assertAllEqual, structure, actual_structure)

        # nested 2-list.
        dtypes = [
            [tf.dtypes.float32, tf.dtypes.float32],
            [tf.dtypes.float32, tf.dtypes.float32],
        ]
        structure = [
            [
                tf.fill([10, 5, 10], tf.cast(-12, tf.dtypes.float32)),
                tf.fill([10], tf.cast(100, tf.dtypes.float32)),
            ],
            [
                tf.fill([10, 5, 10], tf.cast(-12, tf.dtypes.float32)),
                tf.fill([10], tf.cast(100, tf.dtypes.float32)),
            ],
        ]

        with tf.name_scope("NestedList"):
            r = distributed_ops.Value(dtypes)

        r.set(structure)
        actual_structure = r.get()
        tf.nest.map_structure(self.assertAllEqual, structure, actual_structure)

    def testSetGetDict(self):
        # 1-dict.
        dtypes = {"a": tf.dtypes.float32}
        structure = {"a": tf.fill([10, 5, 10], tf.cast(-12, tf.dtypes.float32))}

        with tf.name_scope("Dict"):
            r = distributed_ops.Value(dtypes)

        r.set(structure)
        actual_structure = r.get()
        tf.nest.map_structure(self.assertAllEqual, structure, actual_structure)

        # 2-dict.
        dtypes = {"a": tf.dtypes.float32, "b": tf.dtypes.float32}
        structure = {
            "a": tf.fill([10, 5, 10], tf.cast(-12, tf.dtypes.float32)),
            "b": tf.fill([10], tf.cast(100, tf.dtypes.float32)),
        }

        with tf.name_scope("TwoDict"):
            r = distributed_ops.Value(dtypes)

        r.set(structure)
        actual_structure = r.get()
        tf.nest.map_structure(self.assertAllEqual, structure, actual_structure)

        # nested 2-dict.
        dtypes = {
            "a": {"a": tf.dtypes.float32, "b": tf.dtypes.float32},
            "b": {"a": tf.dtypes.float32, "b": tf.dtypes.float32},
        }
        structure = {
            "a": {
                "a": tf.fill([10, 5, 10], tf.cast(-12, tf.dtypes.float32)),
                "b": tf.fill([10], tf.cast(100, tf.dtypes.float32)),
            },
            "b": {
                "a": tf.fill([10, 5, 10], tf.cast(-12, tf.dtypes.float32)),
                "b": tf.fill([10], tf.cast(100, tf.dtypes.float32)),
            },
        }

        with tf.name_scope("NestedDict"):
            r = distributed_ops.Value(dtypes)

        r.set(structure)
        actual_structure = r.get()
        tf.nest.map_structure(self.assertAllEqual, structure, actual_structure)


if __name__ == "__main__":
    tf.test.main()
