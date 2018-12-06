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
from pyoneer.math import math_ops as pmath_ops

from trfl import action_value_ops


class QLambdaLoss(collections.namedtuple(
    'QLambdaLoss', [
        'value_loss', 'total_loss'])):
    pass


class QLambdaAgent(agent_impl.Agent):
    """Q(λ) algorithm implementation.

    Computes the Q(λ) estimation.

    Reference:
        See "Reinforcement Learning: An Introduction" by Sutton and Barto.
            (http://incompleteideas.net/book/ebook/node78.html).
        and https://github.com/deepmind/trfl
    """

    def __init__(self, value, optimizer):
        """Creates a new QLambdaAgent.

        Args:
            value: the target value to optimize.
            optimizer: Instance of `tf.train.Optimizer`.
        """
        super(QLambdaAgent, self).__init__(optimizer)

        self.value = value

        self.value_loss = array_ops.constant(0.)
        self.total_loss = array_ops.constant(0.)

    @property
    def trainable_variables(self):
        return self.value.trainable_variables

    @property
    def loss(self):
        """Access recent losses computed after `compute_loss(...)` is called.

        Returns:
            a tuple containing `(value_loss, total_loss)`
        """
        return QLambdaLoss(
            value_loss=self.value_loss,
            total_loss=self.total_loss)

    def compute_loss(self, 
                     states, 
                     next_states, 
                     actions, 
                     rewards, 
                     weights, 
                     decay=.999, 
                     lambda_=1.,
                     baseline_scale=1.,
                     **kwargs):
        """Implements Peng's and Watkins' Q(lambda) loss.

        This function is general enough to implement both Peng's and Watkins'
        Q-lambda algorithms.

        See "Reinforcement Learning: An Introduction" by Sutton and Barto.
            (http://incompleteideas.net/book/ebook/node78.html).

        Args:
            states: Tensor of `[B, T, ...]` containing states.
            next_states: Tensor of `[B, T, ...]` containing states[t+1].
            actions: Tensor of `[B, T, ...]` containing actions.
            rewards: Tensor of `[B, T]` containing rewards.
            weights: Tensor of shape `[B, T]` containing weights (1. or 0.).
            decay: scalar or Tensor of shape `[B, T]` containing decays/discounts.
            lambda_: scalar or Tensor of shape `[B, T]` containing q(lambda) parameter.
            baseline_scale: scalar or Tensor of shape `[B, T]` containing the baseline loss scale.
            **kwargs: positional arguments (unused)

        Returns:
            the total loss Tensor of shape [].
        """
        del kwargs
        actions = gen_array_ops.reshape(actions, [states.shape[0], -1])
        sequence_length = math_ops.reduce_sum(weights, axis=1)
        mask = array_ops.expand_dims(
            array_ops.sequence_mask(
                gen_math_ops.maximum(sequence_length - 1, 0), 
                maxlen=states.shape[1], 
                dtype=dtypes.float32), 
            axis=-1)

        pcontinues = decay * weights
        action_values = self.value(states, training=True) * mask
        next_action_values = gen_array_ops.stop_gradient(
            self.value(next_states, training=True))

        lambda_ = gen_array_ops.broadcast_to(lambda_, array_ops.shape(rewards))

        baseline_loss = action_value_ops.qlambda(
            parray_ops.swap_time_major(action_values), 
            parray_ops.swap_time_major(actions), 
            parray_ops.swap_time_major(rewards), 
            parray_ops.swap_time_major(pcontinues), 
            parray_ops.swap_time_major(next_action_values), 
            parray_ops.swap_time_major(lambda_)).loss

        self.value_loss = baseline_scale * losses_impl.compute_weighted_loss(
            baseline_loss, 
            parray_ops.swap_time_major(weights))

        self.total_loss = self.value_loss
        return self.total_loss
