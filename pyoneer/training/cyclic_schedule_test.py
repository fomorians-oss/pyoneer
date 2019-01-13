from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.contrib.eager as tfe

from tensorflow.python.eager import context
from tensorflow.python.platform import test

from pyoneer.training.cyclic_schedule_impl import CyclicSchedule


class CyclicScheduleTest(test.TestCase):
    def test_cyclic_schedule(self):
        with context.eager_mode():
            step = tfe.Variable(0, trainable=False)
            cyclic_lr = CyclicSchedule(
                minval=0.2, maxval=0.4, step_size=50, global_step=step)
            # start of a cycle
            step.assign(1)
            self.assertAllClose(cyclic_lr(), 0.2)
            # apex of the cycle
            step.assign(51)
            self.assertAllClose(cyclic_lr(), 0.4)
            # midway from the apex to the end of the cycle
            step.assign(76)
            self.assertAllClose(cyclic_lr(), 0.2)
            # start of a new cycle
            step.assign(101)
            self.assertAllClose(cyclic_lr(), 0.2)


if __name__ == '__main__':
    test.main()
