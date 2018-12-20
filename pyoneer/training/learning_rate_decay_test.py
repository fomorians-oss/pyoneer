from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.contrib.eager as tfe

from tensorflow.python.eager import context
from tensorflow.python.platform import test

from pyoneer.training import learning_rate_decay

class LRDecayTest(test.TestCase):
    def test_cyclic_lr_scheduler(self):
        with context.eager_mode():
            step = tfe.Variable(0, trainable=False)
            cyclic_lr = learning_rate_decay.cyclic_lr_scheduler(
                lr_min=0.2, lr_max=0.4, step_size=50, global_step=step
            )
            # Start of a cycle
            step.assign(1)
            self.assertAllClose(0.2, cyclic_lr(), 1e-6)
            # Apex of the cycle
            step.assign(51)
            self.assertAllClose(0.4, cyclic_lr(), 1e-6)
            # Midway from the apex to the end of the cycle
            step.assign(76)
            self.assertAllClose(0.2, cyclic_lr(), 1e-6)
            # Start of a new cycle
            step.assign(101)
            self.assertAllClose(0.2, cyclic_lr(), 1e-6)

    def test_cyclicschedule_class(self):
        with context.eager_mode():
            step = tfe.Variable(0, trainable=False)
            cyclic_lr = learning_rate_decay.CyclicSchedule(
                vmin=0.2, vmax=0.4, step_size=50, global_step=step
            )
            # Start of a cycle
            step.assign(1)
            self.assertAllClose(0.2, cyclic_lr(), 1e-6)
            # Apex of the cycle
            step.assign(51)
            self.assertAllClose(0.4, cyclic_lr(), 1e-6)
            # Midway from the apex to the end of the cycle
            step.assign(76)
            self.assertAllClose(0.2, cyclic_lr(), 1e-6)
            # Start of a new cycle
            step.assign(101)
            self.assertAllClose(0.2, cyclic_lr(), 1e-6)

if __name__ == '__main__':
    test.main()
