from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import context
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test
from tensorflow.losses import compute_weighted_loss

from pyoneer.losses import vanilla_policy_gradient_loss

class LossTest(test.TestCase):

    def test_vanilla_policy_gradient_loss(self):
        with context.eager_mode():

            probs = array_ops.constant([.9, .8, .8, .8])
            log_probs = math_ops.log(probs)

            advantages = array_ops.constant([1.,0.,1.,0.])

            loss = vanilla_policy_gradient_loss(log_probs, advantages)
            expected = compute_weighted_loss(advantages * -log_probs)

            self.assertAllClose(loss, expected)

if __name__ == "__main__":
    test.main()
