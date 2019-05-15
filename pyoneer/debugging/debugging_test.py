from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensorflow.python.platform import test

from pyoneer.debugging.debugging_impl import Stopwatch


class DebuggingTest(test.TestCase):
    def test_stopwatch(self):
        with Stopwatch() as stopwatch:
            pass
        self.assertIsNotNone(stopwatch.start_time)
        self.assertIsNotNone(stopwatch.end_time)
        self.assertIsNotNone(stopwatch.duration)


if __name__ == "__main__":
    test.main()
