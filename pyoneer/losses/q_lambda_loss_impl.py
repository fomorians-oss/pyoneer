from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from trfl import indexing_ops
from trfl import sequence_ops

def q_lambda_loss(action_values, 
                  actions, 
                  rewards, 
                  pcontinues, 
                  next_action_values, lambda_):
  """Implements Peng's and Watkins' Q(lambda) loss.

  Reinforcement Learning: An Introduction" by Sutton and Barto.
  (http://incompleteideas.net/book/ebook/node78.html).
  
  Args:
    action_values: `Tensor` holding a sequence of Q-values starting at the first
      timestep; shape `[T, B, num_actions]`
    actions: `Tensor` holding a sequence of action indices, shape `[T, B]`
    rewards: Tensor holding a sequence of rewards, shape `[T, B]`
    pcontinues: `Tensor` holding a sequence of pcontinue values, shape `[T, B]`
    next_action_values: `Tensor` holding a sequence of Q-values for second timestep;
      shape `[T, B, num_actions]`. In a target network setting,
      this quantity is often supplied by the target network.
    lambda_: a scalar or `Tensor` of shape `[T, B]`
      specifying the ratio of mixing between bootstrapped and MC returns;
      if lambda_ is the same for all time steps then the function implements
      Peng's Q-learning algorithm; if lambda_ = 0 at every sub-optimal action
      and a constant otherwise, then the function implements Watkins'
      Q-learning algorithm. Generally lambda_ can be a Tensor of any values
      in the range [0, 1] supplied by the user.
    name: a name of the op.
  Returns:
    a tensor containing the batch of losses, shape `[T, B]`.
  """

    state_values = tf.reduce_max(next_action_values, axis=2)

    target = sequence_ops.multistep_forward_view(
            rewards, pcontinues, state_values, lambda_, back_prop=False)
    
    target = tf.stop_gradient(target)
    
    qa_tm1 = indexing_ops.batched_index(action_values, actions)

    td_error = target - qa_tm1
    loss = 0.5 * tf.square(td_error)
    return loss
