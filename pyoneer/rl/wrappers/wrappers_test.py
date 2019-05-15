from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.platform import test

from pyoneer.rl.wrappers import ObservationCoordinates, ObservationNormalization


class WrappersTest(test.TestCase):
    def test_observation_coords(self):
        self.assertAllEqual(False, True)

    def test_observation_norm(self):
        self.assertAllEqual(False, True)


if __name__ == "__main__":
    test.main()
