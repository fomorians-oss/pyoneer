from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from trfl import indexing_ops
from trfl import sequence_ops

from tensorflow.python.eager import context
from tensorflow.python.platform import test

from pyoneer.losses import q_lambda_loss

class LossTest(test.TestCase):

    def test_q_lambda_loss(self):
        with context.eager_mode():

            action_values = tf.constant(
                [[[1.1, 2.1], [2.1, 3.1]],
                 [[-1.1, 1.1], [-1.1, 0.1]],
                 [[3.1, -3.1], [-2.1, -1.1]]])

            actions =  tf.constant(
                [[0, 1],
                 [1, 0],
                 [0, 0]])

            rewards = tf.constant(
                [[-1.3, 1.3],
                 [-1.3, 5.3],
                 [2.3, -3.3]])

            pcontinues = tf.constant(
                [[0.00, 0.88],
                 [0.89, 1.00],
                 [0.85, 0.83]])

            next_action_values = tf.constant(
                [[[1.2, 2.2], [4.2, 2.2]],
                 [[-1.2, 0.2], [1.2, 1.2]],
                 [[2.2, -1.2], [-1.2, -2.2]]])

            lambda_ = tf.constant(
                [[0.67, 0.68],
                 [0.65, 0.69],
                 [0.66, 0.64]])

            loss = q_lambda_loss(
                action_values,
                actions,
                rewards, 
                pcontinues, 
                next_action_values, 
                lambda_)

            state_values = tf.reduce_max(next_action_values, axis=2)

            target = sequence_ops.multistep_forward_view(
                    rewards, pcontinues, state_values, lambda_, back_prop=False)
            
            target = tf.stop_gradient(target)
            
            qa_tm1 = indexing_ops.batched_index(action_values, actions)

            td_error = target - qa_tm1

            expected = 0.5 * tf.square(td_error)

            self.assertAllClose(loss, expected)

if __name__ == "__main__":
    test.main()
