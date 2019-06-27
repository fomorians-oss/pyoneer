from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from pyoneer.schedules.cyclic_schedule_impl import CyclicSchedule


class CyclicScheduleTest(tf.test.TestCase):
    def test_cyclic_schedule(self):
        cyclic_lr = CyclicSchedule(
            minval=0.2, maxval=0.4, step_size=500, decay_to_zero=False
        )
        # start of a cycle
        self.assertAllClose(cyclic_lr(0), 0.2)
        # apex of the cycle
        self.assertAllClose(cyclic_lr(500), 0.4)
        # midway from the apex to the end of a cycle
        self.assertAllClose(cyclic_lr(750), 0.3)
        # end of a cycle
        self.assertAllClose(cyclic_lr(999), 0.2004)
        # start of a new cycle
        self.assertAllClose(cyclic_lr(1000), 0.2)

    def test_cyclic_schedule_decay(self):
        cyclic_lr = CyclicSchedule(
            minval=0.2, maxval=0.4, step_size=500, decay_to_zero=True
        )
        # start of a cycle
        self.assertAllClose(cyclic_lr(0), 0.2)
        # apex of the cycle
        self.assertAllClose(cyclic_lr(500), 0.4)
        # midway from the apex to the end of a cycle
        self.assertAllClose(cyclic_lr(750), 0.2)
        # end of a cycle
        self.assertAllClose(cyclic_lr(999), 0.0008)
        # start of a new cycle
        self.assertAllClose(cyclic_lr(1000), 0.2)


if __name__ == "__main__":
    tf.test.main()
