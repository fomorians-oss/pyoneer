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
from pyoneer.manip import array_ops as parray_ops

from trfl import action_value_ops


class QLoss(collections.namedtuple(
    'QLoss', [
        'value_loss', 'total_loss'])):
    pass


class QAgent(agent_impl.Agent):
    """Q algorithm implementation.

    Computes the Q estimation:
    """

    def __init__(self, value, optimizer):
        super(QAgent, self).__init__(optimizer)

        self.value = value

        self.value_loss = array_ops.constant(0.)
        self.total_loss = array_ops.constant(0.)

    @property
    def trainable_variables(self):
        return self.value.trainable_variables

    @property
    def loss(self):
        return QLoss(
            value_loss=self.value_loss,
            total_loss=self.total_loss)

    def compute_loss(self, rollouts, delay=.999):
        """Implements the Q-learning loss.

        The loss is `0.5` times the squared difference between `q_tm1[a_tm1]` and
        the target `r_t + pcont_t * max q_t`.

        See "Reinforcement Learning: An Introduction" by Sutton and Barto.
            (http://incompleteideas.net/book/ebook/node65.html).
        """
        pcontinues = delay * rollouts.weights
        action_values = self.value(rollouts.states, training=True)
        next_action_values = self.value(rollouts.next_states, training=True)

        self.value_loss = math_ops.reduce_mean(
            action_value_ops.qlearning(
                action_values, 
                rollouts.actions, 
                rollouts.rewards, 
                pcontinues, 
                next_action_values).loss,
            axis=0)

        self.total_loss = self.value_loss
        return self.total_loss
