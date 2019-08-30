from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import tensorflow as tf


class Unroll(collections.namedtuple(
        'Unroll', ['outputs', 'initializer', 'initial_time_step'])):
    pass


def n_step_unroll(n_step,
                  initializer,
                  agent_step_fn,
                  env_step_fn,
                  initial_time_step=0):
    """Unroll interactions for n-steps.

    Args:
        n_step: Tensor of shape [].
        initializer: Tuple containing initial values:
            (agent_output, env_output)
        agent_step_fn: Agent step function:
            fn(env_output, agent_output, time_step)
        env_step_fn: Environment step function:
            fn(agent_output, env_output, time_step)
        initial_time_step: The initial step value.

    Returns:
        Tuple containing:
            - Tensors of shape [Time x ...] with same structure as
                `initializer`.
            - Tuple containing final values:
                (agent_output, env_output)
            - Final time step.
    """
    agent_output, env_output = initializer

    def create_ta(t):
        return tf.TensorArray(t.dtype, size=n_step, element_shape=t.shape)

    # Create arrays.
    agent_output_tas = tf.nest.map_structure(create_ta, agent_output)
    env_output_tas = tf.nest.map_structure(create_ta, env_output)
    tas = agent_output_tas, env_output_tas

    def cond_fn(tas, outputs, time_step, index):
        _, env_output = outputs
        return tf.reduce_all(~env_output.terminal)

    def step_fn(tas, outputs, time_step, index):
        """Compute the next transition tuple.

            a[t]   = agent(s[t], a[t-1], t)
            s[t+1] = env(a[t], s[t], t)
                   => {state, action, reward, next_state, terminal}
        """
        agent_output, env_output = outputs

        agent_output = agent_step_fn(env_output, agent_output, time_step)
        env_output = env_step_fn(agent_output, env_output, time_step)

        def write_ta(ta, value):
            return ta.write(index, value)

        agent_output_tas, env_output_tas = tas
        agent_output_tas = tf.nest.map_structure(write_ta, agent_output_tas,
                                                 agent_output)
        env_output_tas = tf.nest.map_structure(write_ta, env_output_tas,
                                               env_output)

        return ((agent_output_tas, env_output_tas),
                (agent_output, env_output),
                time_step + 1, index + 1)

    # Unroll for n_step [T x ...]
    tas, finalizer, final_time_step, _ = tf.while_loop(
        cond_fn, step_fn, (tas, initializer, initial_time_step,
                           tf.constant(0, tf.int32)),
        parallel_iterations=1, maximum_iterations=n_step)

    def stack_ta(ta):
        return ta.stack()

    # Stack arrays into tensors.
    agent_output_tas, env_output_tas = tas
    agent_output = tf.nest.map_structure(stack_ta, agent_output_tas)
    env_output = tf.nest.map_structure(stack_ta, env_output_tas)

    return Unroll(
        outputs=(agent_output, env_output),
        initializer=finalizer,
        initial_time_step=final_time_step)


class Rollout(object):

    def __init__(self, env, agent, n_step, time_major=False):
        """Creates a new Rollout.

        Nest-compatible API for arbitrarily nested transitions.

        Args:
            env: The Env interface.
            agent: The Agent interface.
            n_step: The number of steps to rollout.
            time_major: Time major outputs.
        """
        self._env = env
        self._agent = agent
        self._n_step = n_step
        self._time_major = time_major

    @tf.function
    def __call__(self, initializer=None, initial_time_step=0):
        """Rollout and return trajectories.

        Args:
            initializer: (Optional) The initial inputs to the rollout.
            initial_time_step: (Optional) The initial time step.

        Returns:
            (outputs, initializer, initial_time_step)
        """
        if initializer is None:
            env_output = self._env.reset()
            agent_output = self._agent.reset(env_output)
            initializer = (agent_output, env_output)

        unroll = n_step_unroll(
            self._n_step,
            initializer,
            self._agent.step,
            self._env.step,
            initial_time_step=initial_time_step)

        if self._time_major:
            return unroll

        return unroll._replace(
            outputs=tf.nest.map_structure(
                lambda t: tf.transpose(t, [1, 0] + list(range(2, t.shape.rank))),
                unroll.outputs))
