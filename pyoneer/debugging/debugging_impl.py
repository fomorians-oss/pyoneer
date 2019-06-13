from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class Stopwatch(object):
    """
    Stopwatch for measuring how long operations take. Great for fast and easy profiling.

    Example:
    >>> x = tf.constant(1.0)
    >>> y = tf.constant(2.0)
    >>> with Stopwatch() as watch:
    >>>    z = x + y
    >>> tf.print(watch.duration)
    >>> # 0.00021505355834960938
    """

    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.duration = None

    def start(self):
        self.start_time = tf.timestamp()

    def stop(self):
        self.end_time = tf.timestamp()
        self.duration = self.end_time - self.start_time

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()
