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
from pyoneer.rl.agents import q_agent_impl
from pyoneer.features import array_ops as parray_ops

from trfl import action_value_ops


class DoubleQAgent(agent_impl.Agent):
    """Double Q algorithm implementation.

    Computes the Double Q estimation:
    """

    def __init__(self, value, target_value, optimizer):
        super(DoubleQAgent, self).__init__(optimizer)

        self.value = value
        self.target_value = target_value

        self.value_loss = array_ops.constant(0.)
        self.total_loss = array_ops.constant(0.)

    @property
    def trainable_variables(self):
        return self.value.trainable_variables

    @property
    def loss(self):
        return q_agent_impl.QLoss(
            value_loss=self.value_loss,
            total_loss=self.total_loss)

    def compute_loss(self, rollouts, delay=.999):
        """Implements the double Q-learning loss.
        
        The loss is `0.5` times the squared difference between `q_tm1[a_tm1]` and
        the target `r_t + pcont_t * q_t_value[argmax q_t_selector]`.
        
        See "Double Q-learning" by van Hasselt.
            (https://papers.nips.cc/paper/3964-double-q-learning.pdf).
        """
        pcontinues = delay * rollouts.weights
        action_values = self.value(rollouts.states, training=True)
        next_action_values = self.value(rollouts.next_states, training=True)

        self.value_loss = math_ops.reduce_mean(
            action_value_ops.double_qlearning(
                action_values, 
                rollouts.actions, 
                rollouts.rewards, 
                pcontinues, 
                next_action_values,
                next_action_values).loss,
            axis=0)

        self.total_loss = self.value_loss
        return self.total_loss
