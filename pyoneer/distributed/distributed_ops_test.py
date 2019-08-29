from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import threading
try:
    import redis
except ImportError as e:
    print(('Redis must be installed to use `pynr.distributed`. '
           'Run `pip install redis`.'))
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
        structure = (tf.fill([10, 5, 10], tf.cast(-12, tf.dtypes.float32)),
                     tf.fill([10], tf.cast(100, tf.dtypes.float32)))

        codec = distributed_ops.TensorCodec(dtypes=dtypes)
        encoded_msg = codec.encode(structure)
        decoded = codec.decode(encoded_msg)

        tf.nest.map_structure(self.assertAllEqual, decoded, structure)

        # nested 2-tuple.
        dtypes = ((tf.dtypes.float32, tf.dtypes.float32),
                  (tf.dtypes.float32, tf.dtypes.float32))
        structure = ((tf.fill([10, 5, 10], tf.cast(-12, tf.dtypes.float32)),
                     tf.fill([10], tf.cast(100, tf.dtypes.float32))),
                     (tf.fill([10, 5, 10], tf.cast(-12, tf.dtypes.float32)),
                     tf.fill([10], tf.cast(100, tf.dtypes.float32))))

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
        structure = [tf.fill([10, 5, 10], tf.cast(-12, tf.dtypes.float32)),
                     tf.fill([10], tf.cast(100, tf.dtypes.float32))]

        codec = distributed_ops.TensorCodec(dtypes=dtypes)
        encoded_msg = codec.encode(structure)
        decoded = codec.decode(encoded_msg)

        tf.nest.map_structure(self.assertAllEqual, decoded, structure)

        # nested 2-list.
        dtypes = [[tf.dtypes.float32, tf.dtypes.float32],
                  [tf.dtypes.float32, tf.dtypes.float32]]
        structure = [[tf.fill([10, 5, 10], tf.cast(-12, tf.dtypes.float32)),
                     tf.fill([10], tf.cast(100, tf.dtypes.float32))],
                     [tf.fill([10, 5, 10], tf.cast(-12, tf.dtypes.float32)),
                     tf.fill([10], tf.cast(100, tf.dtypes.float32))]]

        codec = distributed_ops.TensorCodec(dtypes=dtypes)
        encoded_msg = codec.encode(structure)
        decoded = codec.decode(encoded_msg)

        tf.nest.map_structure(self.assertAllEqual, decoded, structure)

    def testDict(self):
        # 1-dict.
        dtypes = {'a': tf.dtypes.float32}
        structure = {'a': tf.fill([10, 5, 10],
                                  tf.cast(-12, tf.dtypes.float32))}

        codec = distributed_ops.TensorCodec(dtypes=dtypes)
        encoded_msg = codec.encode(structure)
        decoded = codec.decode(encoded_msg)

        tf.nest.map_structure(self.assertAllEqual, decoded, structure)

        # 2-dict.
        dtypes = {'a': tf.dtypes.float32, 'b': tf.dtypes.float32}
        structure = {'a': tf.fill([10, 5, 10],
                                  tf.cast(-12, tf.dtypes.float32)),
                     'b': tf.fill([10], tf.cast(100, tf.dtypes.float32))}

        codec = distributed_ops.TensorCodec(dtypes=dtypes)
        encoded_msg = codec.encode(structure)
        decoded = codec.decode(encoded_msg)

        tf.nest.map_structure(self.assertAllEqual, decoded, structure)

        # nested 2-dict.
        dtypes = {'a': {'a': tf.dtypes.float32, 'b': tf.dtypes.float32},
                  'b': {'a': tf.dtypes.float32, 'b': tf.dtypes.float32}}
        structure = {'a': {'a': tf.fill([10, 5, 10],
                                        tf.cast(-12, tf.dtypes.float32)),
                           'b': tf.fill([10],
                                        tf.cast(100, tf.dtypes.float32))},
                     'b': {'a': tf.fill([10, 5, 10],
                                        tf.cast(-12, tf.dtypes.float32)),
                           'b': tf.fill([10],
                                        tf.cast(100, tf.dtypes.float32))}}

        codec = distributed_ops.TensorCodec(dtypes=dtypes)
        encoded_msg = codec.encode(structure)
        decoded = codec.decode(encoded_msg)

        tf.nest.map_structure(self.assertAllEqual, decoded, structure)


class QueueTest(tf.test.TestCase):

    def testEnqueueDequeueTuple(self):
        redis_host = '127.0.0.1'
        redis_port = 6380
        q_key = 'q'
        srvr = redis.Redis(host=redis_host,
                           port=redis_port,
                           db=0)

        # 1-tuple.
        dtypes = (tf.dtypes.float32,)
        structure = (tf.fill([10], tf.cast(100, tf.dtypes.float32)),)

        q = distributed_ops.Queue(srvr, q_key, dtypes)
        q.enqueue(structure)
        actual_structure = q.dequeue()
        tf.nest.map_structure(self.assertAllEqual, structure, actual_structure)

        # 2-tuple.
        dtypes = (tf.dtypes.float32, tf.dtypes.float32)
        structure = (tf.fill([10, 5, 10], tf.cast(-12, tf.dtypes.float32)),
                     tf.fill([10], tf.cast(100, tf.dtypes.float32)))

        q = distributed_ops.Queue(srvr, q_key, dtypes)
        q.enqueue(structure)
        actual_structure = q.dequeue()
        tf.nest.map_structure(self.assertAllEqual, structure, actual_structure)

        # nested tuple.
        dtypes = ((tf.dtypes.float32, tf.dtypes.float32),
                  (tf.dtypes.float32, tf.dtypes.float32))
        structure = ((tf.fill([10, 5, 10], tf.cast(-12, tf.dtypes.float32)),
                     tf.fill([10], tf.cast(100, tf.dtypes.float32))),
                     (tf.fill([10, 5, 10], tf.cast(-12, tf.dtypes.float32)),
                     tf.fill([10], tf.cast(100, tf.dtypes.float32))))

        q = distributed_ops.Queue(srvr, q_key, dtypes)
        q.enqueue(structure)
        actual_structure = q.dequeue()
        tf.nest.map_structure(self.assertAllEqual, structure, actual_structure)

        # dequeue many
        q = distributed_ops.Queue(srvr, q_key, dtypes)
        for many, axis in zip([1, 2, 10], [0, 1, -1]):
            for _ in range(many):
                q.enqueue(structure)
            actual_structure = q.dequeue_many(many, axis=axis)

            def equal_stacked(a, bs):
                for b in tf.unstack(bs, axis=axis):
                    self.assertAllEqual(a, b)

            tf.nest.map_structure(equal_stacked, structure, actual_structure)

    def testEnqueueDequeueList(self):
        redis_host = '127.0.0.1'
        redis_port = 6380
        q_key = 'q'
        srvr = redis.Redis(host=redis_host,
                           port=redis_port,
                           db=0)

        # 1-list.
        dtypes = [tf.dtypes.float32]
        structure = [tf.fill([10], tf.cast(100, tf.dtypes.float32))]

        q = distributed_ops.Queue(srvr, q_key, dtypes)
        q.enqueue(structure)
        actual_structure = q.dequeue()
        tf.nest.map_structure(self.assertAllEqual, structure, actual_structure)

        # 2-list.
        dtypes = [tf.dtypes.float32, tf.dtypes.float32]
        structure = [tf.fill([10, 5, 10], tf.cast(-12, tf.dtypes.float32)),
                     tf.fill([10], tf.cast(100, tf.dtypes.float32))]

        q = distributed_ops.Queue(srvr, q_key, dtypes)
        q.enqueue(structure)
        actual_structure = q.dequeue()
        tf.nest.map_structure(self.assertAllEqual, structure, actual_structure)

        # nested 2-list.
        dtypes = [[tf.dtypes.float32, tf.dtypes.float32],
                  [tf.dtypes.float32, tf.dtypes.float32]]
        structure = [[tf.fill([10, 5, 10], tf.cast(-12, tf.dtypes.float32)),
                     tf.fill([10], tf.cast(100, tf.dtypes.float32))],
                     [tf.fill([10, 5, 10], tf.cast(-12, tf.dtypes.float32)),
                     tf.fill([10], tf.cast(100, tf.dtypes.float32))]]

        q = distributed_ops.Queue(srvr, q_key, dtypes)
        q.enqueue(structure)
        actual_structure = q.dequeue()
        tf.nest.map_structure(self.assertAllEqual, structure, actual_structure)

        # dequeue many
        q = distributed_ops.Queue(srvr, q_key, dtypes)
        for many, axis in zip([1, 2, 10], [0, 1, -1]):
            for _ in range(many):
                q.enqueue(structure)
            actual_structure = q.dequeue_many(many, axis=axis)

            def equal_stacked(a, bs):
                for b in tf.unstack(bs, axis=axis):
                    self.assertAllEqual(a, b)

            tf.nest.map_structure(equal_stacked, structure, actual_structure)

    def testEnqueueDequeueDict(self):
        redis_host = '127.0.0.1'
        redis_port = 6380
        q_key = 'q'
        srvr = redis.Redis(host=redis_host,
                           port=redis_port,
                           db=0)

        # 1-dict.
        dtypes = {'a': tf.dtypes.float32}
        structure = {'a': tf.fill([10, 5, 10],
                                  tf.cast(-12, tf.dtypes.float32))}

        q = distributed_ops.Queue(srvr, q_key, dtypes)
        q.enqueue(structure)
        actual_structure = q.dequeue()
        tf.nest.map_structure(self.assertAllEqual, structure, actual_structure)

        # 2-dict.
        dtypes = {'a': tf.dtypes.float32, 'b': tf.dtypes.float32}
        structure = {'a': tf.fill([10, 5, 10],
                                  tf.cast(-12, tf.dtypes.float32)),
                     'b': tf.fill([10], tf.cast(100, tf.dtypes.float32))}

        q = distributed_ops.Queue(srvr, q_key, dtypes)
        q.enqueue(structure)
        actual_structure = q.dequeue()
        tf.nest.map_structure(self.assertAllEqual, structure, actual_structure)

        # nested 2-dict.
        dtypes = {'a': {'a': tf.dtypes.float32, 'b': tf.dtypes.float32},
                  'b': {'a': tf.dtypes.float32, 'b': tf.dtypes.float32}}
        structure = {'a': {'a': tf.fill([10, 5, 10],
                                        tf.cast(-12, tf.dtypes.float32)),
                           'b': tf.fill([10],
                                        tf.cast(100, tf.dtypes.float32))},
                     'b': {'a': tf.fill([10, 5, 10],
                                        tf.cast(-12, tf.dtypes.float32)),
                           'b': tf.fill([10],
                                        tf.cast(100, tf.dtypes.float32))}}

        q = distributed_ops.Queue(srvr, q_key, dtypes)
        q.enqueue(structure)
        actual_structure = q.dequeue()
        tf.nest.map_structure(self.assertAllEqual, structure, actual_structure)

        # dequeue many
        q = distributed_ops.Queue(srvr, q_key, dtypes)
        for many, axis in zip([1, 2, 10], [0, 1, -1]):
            for _ in range(many):
                q.enqueue(structure)
            actual_structure = q.dequeue_many(many, axis=axis)

            def equal_stacked(a, bs):
                for b in tf.unstack(bs, axis=axis):
                    self.assertAllEqual(a, b)

            tf.nest.map_structure(equal_stacked, structure, actual_structure)


class ConditionTest(tf.test.TestCase):

    def testWaitNotifyAll(self):
        redis_host = '127.0.0.1'
        redis_port = 6380
        c_key = 'c'
        num_consumers = 5
        srvr = redis.Redis(host=redis_host,
                           port=redis_port,
                           db=0)

        condition = distributed_ops.Condition(srvr, c_key)

        def consumer_fn(cond, w_id):
            cond.wait(w_id)

        consumers = []
        for _ in range(num_consumers):
            consumer = threading.Thread(target=consumer_fn,
                                        args=(condition, 0))
            consumer.start()
            consumers.append(consumer)
        time.sleep(1)

        condition.notify_all()
        for consumer in consumers:
            consumer.join()


class RegisterTest(tf.test.TestCase):

    def testSetGetTuple(self):
        redis_host = '127.0.0.1'
        redis_port = 6380
        r_key = 'r'
        srvr = redis.Redis(host=redis_host,
                           port=redis_port,
                           db=0)

        # 1-tuple.
        dtypes = (tf.dtypes.float32,)
        structure = (tf.fill([10], tf.cast(100, tf.dtypes.float32)),)

        r = distributed_ops.Register(srvr, r_key, dtypes)
        r.set(structure)
        actual_structure = r.get()
        tf.nest.map_structure(self.assertAllEqual, structure, actual_structure)

        # 2-tuple.
        dtypes = (tf.dtypes.float32, tf.dtypes.float32)
        structure = (tf.fill([10, 5, 10], tf.cast(-12, tf.dtypes.float32)),
                     tf.fill([10], tf.cast(100, tf.dtypes.float32)))

        r = distributed_ops.Register(srvr, r_key, dtypes)
        r.set(structure)
        actual_structure = r.get()
        tf.nest.map_structure(self.assertAllEqual, structure, actual_structure)

        # nested tuple.
        dtypes = ((tf.dtypes.float32, tf.dtypes.float32),
                  (tf.dtypes.float32, tf.dtypes.float32))
        structure = ((tf.fill([10, 5, 10], tf.cast(-12, tf.dtypes.float32)),
                     tf.fill([10], tf.cast(100, tf.dtypes.float32))),
                     (tf.fill([10, 5, 10], tf.cast(-12, tf.dtypes.float32)),
                     tf.fill([10], tf.cast(100, tf.dtypes.float32))))

        r = distributed_ops.Register(srvr, r_key, dtypes)
        r.set(structure)
        actual_structure = r.get()
        tf.nest.map_structure(self.assertAllEqual, structure, actual_structure)

    def testSetGetList(self):
        redis_host = '127.0.0.1'
        redis_port = 6380
        r_key = 'r'
        srvr = redis.Redis(host=redis_host,
                           port=redis_port,
                           db=0)

        # 1-list.
        dtypes = [tf.dtypes.float32]
        structure = [tf.fill([10], tf.cast(100, tf.dtypes.float32))]

        r = distributed_ops.Register(srvr, r_key, dtypes)
        r.set(structure)
        actual_structure = r.get()
        tf.nest.map_structure(self.assertAllEqual, structure, actual_structure)

        # 2-list.
        dtypes = [tf.dtypes.float32, tf.dtypes.float32]
        structure = [tf.fill([10, 5, 10], tf.cast(-12, tf.dtypes.float32)),
                     tf.fill([10], tf.cast(100, tf.dtypes.float32))]

        r = distributed_ops.Register(srvr, r_key, dtypes)
        r.set(structure)
        actual_structure = r.get()
        tf.nest.map_structure(self.assertAllEqual, structure, actual_structure)

        # nested 2-list.
        dtypes = [[tf.dtypes.float32, tf.dtypes.float32],
                  [tf.dtypes.float32, tf.dtypes.float32]]
        structure = [[tf.fill([10, 5, 10], tf.cast(-12, tf.dtypes.float32)),
                     tf.fill([10], tf.cast(100, tf.dtypes.float32))],
                     [tf.fill([10, 5, 10], tf.cast(-12, tf.dtypes.float32)),
                     tf.fill([10], tf.cast(100, tf.dtypes.float32))]]

        r = distributed_ops.Register(srvr, r_key, dtypes)
        r.set(structure)
        actual_structure = r.get()
        tf.nest.map_structure(self.assertAllEqual, structure, actual_structure)

    def testSetGetDict(self):
        redis_host = '127.0.0.1'
        redis_port = 6380
        r_key = 'r'
        srvr = redis.Redis(host=redis_host,
                           port=redis_port,
                           db=0)

        # 1-dict.
        dtypes = {'a': tf.dtypes.float32}
        structure = {'a': tf.fill([10, 5, 10],
                                  tf.cast(-12, tf.dtypes.float32))}

        r = distributed_ops.Register(srvr, r_key, dtypes)
        r.set(structure)
        actual_structure = r.get()
        tf.nest.map_structure(self.assertAllEqual, structure, actual_structure)

        # 2-dict.
        dtypes = {'a': tf.dtypes.float32, 'b': tf.dtypes.float32}
        structure = {'a': tf.fill([10, 5, 10],
                                  tf.cast(-12, tf.dtypes.float32)),
                     'b': tf.fill([10], tf.cast(100, tf.dtypes.float32))}

        r = distributed_ops.Register(srvr, r_key, dtypes)
        r.set(structure)
        actual_structure = r.get()
        tf.nest.map_structure(self.assertAllEqual, structure, actual_structure)

        # nested 2-dict.
        dtypes = {'a': {'a': tf.dtypes.float32, 'b': tf.dtypes.float32},
                  'b': {'a': tf.dtypes.float32, 'b': tf.dtypes.float32}}
        structure = {'a': {'a': tf.fill([10, 5, 10],
                                        tf.cast(-12, tf.dtypes.float32)),
                           'b': tf.fill([10],
                                        tf.cast(100, tf.dtypes.float32))},
                     'b': {'a': tf.fill([10, 5, 10],
                                        tf.cast(-12, tf.dtypes.float32)),
                           'b': tf.fill([10],
                                        tf.cast(100, tf.dtypes.float32))}}

        r = distributed_ops.Register(srvr, r_key, dtypes)
        r.set(structure)
        actual_structure = r.get()
        tf.nest.map_structure(self.assertAllEqual, structure, actual_structure)


if __name__ == "__main__":
    tf.test.main()
