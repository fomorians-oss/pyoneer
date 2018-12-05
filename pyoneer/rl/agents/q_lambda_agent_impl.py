from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.eager import backprop
from tensorflow.python.ops.losses import losses_impl
from tensorflow.python.training import optimizer

from pyoneer.rl.agents import agent_impl
from pyoneer.manip import array_ops as parray_ops

from trfl import action_value_ops


class QLambdaLoss(collections.namedtuple(
    'QLambdaLoss', [
        'value_loss', 'total_loss'])):
    pass


class QLambdaAgent(agent_impl.Agent):
    """Q(λ) algorithm implementation.

    Computes the Q(λ) estimation:
    """

    def __init__(self, value, optimizer):
        super(QLambdaAgent, self).__init__(optimizer)

        self.value = value

        self.value_loss = array_ops.constant(0.)
        self.total_loss = array_ops.constant(0.)

    @property
    def trainable_variables(self):
        return self.value.trainable_variables

    @property
    def loss(self):
        return QLambdaLoss(
            value_loss=self.value_loss,
            total_loss=self.total_loss)

    def compute_loss(self, rollouts, delay=.999, lambda_=1.):
        """Implements Peng's and Watkins' Q(lambda) loss.

        This function is general enough to implement both Peng's and Watkins'
        Q-lambda algorithms.

        See "Reinforcement Learning: An Introduction" by Sutton and Barto.
            (http://incompleteideas.net/book/ebook/node78.html).
        """
        batch_size = len(rollouts)
        actions = gen_array_ops.reshape(rollouts.actions, [batch_size, -1])
        sequence_length = math_ops.reduce_sum(rollouts.weights, axis=1)
        mask = array_ops.expand_dims(
            array_ops.sequence_mask(
                gen_math_ops.maximum(sequence_length - 1, 0), 
                maxlen=rollouts.states.shape[1], 
                dtype=dtypes.float32), 
            axis=-1)

        pcontinues = delay * rollouts.weights
        action_values = self.value(rollouts.states, training=True) * mask
        next_action_values = gen_array_ops.stop_gradient(
            self.value(rollouts.next_states, training=True))

        lambda_ = gen_array_ops.broadcast_to(lambda_, array_ops.shape(rollouts.rewards))

        self.value_loss = math_ops.reduce_mean(
            action_value_ops.qlambda(
                parray_ops.swap_time_major(action_values), 
                parray_ops.swap_time_major(actions), 
                parray_ops.swap_time_major(rollouts.rewards), 
                parray_ops.swap_time_major(pcontinues), 
                parray_ops.swap_time_major(next_action_values), 
                parray_ops.swap_time_major(lambda_)).loss, 
            axis=0)

        self.total_loss = self.value_loss
        return self.total_loss
