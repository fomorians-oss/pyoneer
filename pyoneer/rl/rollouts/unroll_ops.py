from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import tensorflow as tf


class Unroll(collections.namedtuple(
        'Unroll', ['outputs', 'finalizer'])):
    """Creates a new Unroll.

    Args:
        outputs: The outputs of an unroll, shape
            [Batch x Time x ...] or (time_major) [Time x Batch x ...].
        finalizer: The final outputs of an unroll with shape [Batch x ...].
    """
    __slots__ = ()


class Transition(collections.namedtuple(
        'Transition', ['state', 'reward', 'next_state', 'terminal', 'weight'])):

    """Creates a new Transition.

    Args:
        state: The state for this transition.
        reward: The reward for this transition.
        next_state: The next state for this transition.
        terminal: The terminal condition of this transition.
        weight: The weight assigned to this transition.
    """
    __slots__ = ()


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
        self._n_step = tf.convert_to_tensor(n_step, dtype=tf.dtypes.int32)
        self._time_major = tf.convert_to_tensor(time_major, dtype=tf.dtypes.bool)

        self.output_dtypes = (self._agent.output_dtypes, self._env.output_dtypes)
        self.output_shapes = (self._agent.output_shapes, self._env.output_shapes)
        self.output_specs = (self._agent.output_specs, self._env.output_specs)

    @property
    def env(self):
        return self._env

    @property
    def agent(self):
        return self._agent

    def _reset(self, initializer=None):
        if initializer is None:
            env_output = self._env.reset()
            agent_output = self._agent.reset(env_output)
            initializer = (agent_output, env_output)
            return agent_output, env_output, initializer

        # TODO(wenkesj): Handle auto-reset?
        agent_output, env_output = initializer
        return agent_output, env_output, initializer

    @tf.function
    def __call__(self, initializer=None):
        """Rollout env-agent interaction for n-steps starting from the conditions.

        At each step of the n-step rollout, the environment passes it's output to
            the agent and the agent passes it's output to the environment:

            ```none
            agent_output, env_output = initializer
            agent_output = agent.step(env_output, agent_output)
            env_output = env.step(agent_output, env_output)
            finalizer = agent_output, env_output
            ```

        If the `initializer` argument is `None`, than the `initializer` will be
            set as follows:

            ```none
            env_output = env.reset()
            agent_output = agent.reset(env_output)
            initializer = agent_output, env_output
            ```

        An n-step unroll will terminate early if all `env_output.terminal` are set to
            True.

        Args:
            initializer: (Optional) The initial inputs to the rollout.
                If `None` then the initializer will be replaced with
                `env_output = env.reset()` and `agent.reset(env_output)`.

        Returns:
            (outputs, finalizer)
        """
        agent_output, env_output, initializer = self._reset(initializer)

        def create_ta(t):
            return tf.TensorArray(t.dtype, size=self._n_step, element_shape=t.shape)

        # Create arrays.
        agent_output_tas = tf.nest.map_structure(create_ta, agent_output)
        env_output_tas = tf.nest.map_structure(create_ta, env_output)
        tas = agent_output_tas, env_output_tas

        def cond_fn(tas, outputs, index):
            _, env_output = outputs
            return ~tf.reduce_all(env_output.terminal)

        def step_fn(tas, outputs, index):
            """Compute the next transition tuple.

                a[t]   = agent(s[t], a[t-1], t)
                s[t+1] = env(a[t], s[t], t)
                    => {state, action, reward, next_state, terminal}
            """
            agent_output, env_output = outputs

            agent_output = self._agent.step(env_output, agent_output)
            env_output = self._env.step(agent_output, env_output)

            def write_ta(ta, value):
                return ta.write(index, value)

            agent_output_tas, env_output_tas = tas
            agent_output_tas = tf.nest.map_structure(write_ta, agent_output_tas,
                                                    agent_output)
            env_output_tas = tf.nest.map_structure(write_ta, env_output_tas,
                                                   env_output)

            return ((agent_output_tas, env_output_tas),
                    (agent_output, env_output),
                    index + 1)

        # Unroll for n_step [T x ...]
        tas, finalizer, _ = tf.while_loop(
            cond_fn, step_fn,
            (tas, initializer, tf.constant(0, tf.int32)),
            parallel_iterations=1, maximum_iterations=self._n_step)

        def stack_ta(ta):
            return ta.stack()

        # Stack arrays into tensors.
        agent_output_tas, env_output_tas = tas
        agent_output = tf.nest.map_structure(stack_ta, agent_output_tas)
        env_output = tf.nest.map_structure(stack_ta, env_output_tas)

        unroll = Unroll(
            outputs=(agent_output, env_output),
            finalizer=finalizer)

        if self._time_major:
            return unroll

        return unroll._replace(
            outputs=tf.nest.map_structure(
                lambda t: tf.transpose(t, [1, 0] + list(range(2, t.shape.rank))),
                unroll.outputs))
