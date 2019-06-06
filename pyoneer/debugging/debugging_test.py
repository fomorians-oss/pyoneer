from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from pyoneer.debugging.debugging_impl import Stopwatch


class DebuggingTest(tf.test.TestCase):
    def test_stopwatch(self):
        with Stopwatch() as stopwatch:
            pass
        self.assertIsNotNone(stopwatch.start_time)
        self.assertIsNotNone(stopwatch.end_time)
        self.assertIsNotNone(stopwatch.duration)


if __name__ == "__main__":
    tf.test.main()
