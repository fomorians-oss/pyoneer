from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import collections

from gym import spaces

import tensorflow as tf

from pyoneer.debugging import debugging_ops
from pyoneer.rl.wrappers import batch_impl


class _Counter(object):

    def __init__(self):
        """Creates a new _Counter.

        This is a simple reference counter as an object.
        """
        self.value = 0

    def increment(self):
        self.value += 1
        return self

    def get(self):
        return self.value


def _space_to_spec(space, counter, name):
    """Determines a tf.TensorSpec from the `gym.Space`.

    Args:
        space: a `gym.Space` instance (i.e. `env.action_space`)
        counter: The counter to track the

    Raises:
        `TypeError` when space is not a `gym.Space` instance.

    Returns:
        Possibly nested `tf.TensorSpec`.
    """
    if space.__class__ in [spaces.Discrete, spaces.MultiDiscrete,
                           spaces.MultiBinary, spaces.Box]:
        return tf.TensorSpec(tf.TensorShape(space.shape),
                             tf.dtypes.as_dtype(space.dtype),
                             name='{}_{}'.format(name, counter.get()))
    elif isinstance(space, spaces.Tuple) or isinstance(space, spaces.Dict):
        def to_spec_fn(space):
            return _space_to_spec(space, name, counter.increment())
        return tf.nest.map_structure(to_spec_fn, space.spaces)
    raise TypeError('`space` not supported: {}'.format(type(space)))


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


class Env(object):

    def __init__(self, env):
        """Creates a new Env.

        This is a thin wrapper around `gym.Env` to directly work with `pyrl.Rollout`.

        Args:
            env: An `gym.Env` or `pynr.wrappers.Batch` instance.
        """
        self._env = env

        # Check if the environment is batched.
        self._is_batched = isinstance(env, batch_impl.Batch)
        if self._is_batched:
            self._output_shape_list = [len(env)]
        else:
            self._output_shape_list = [1]

        # Specs.
        self.state_spec = _space_to_spec(env.observation_space, _Counter(), name='State')
        self.next_state_spec = _space_to_spec(env.observation_space, _Counter(), name='NextState')
        self.action_spec = _space_to_spec(env.action_space, _Counter(), name='Action')
        if hasattr(env, 'reward_space'):
            self.reward_spec = _space_to_spec(env.reward_space, _Counter(), name='Reward')
        else:
            self.reward_spec = tf.TensorSpec([], tf.dtypes.float32, name='Reward')
        self.terminal_spec = tf.TensorSpec([], tf.dtypes.bool, name='Terminal')
        self.weight_spec = tf.TensorSpec([], tf.dtypes.float32, name='Weight')

        # TODO(wenkesj): Pass optional initial
        #   (reward, state, terminal, and weight).
        (self._initial_reward,
         self._initial_state,
         self._initial_terminal,
         self._initial_weight) = debugging_ops.mock_spec(
            tf.TensorShape(self._output_shape_list),
            (self.reward_spec, self.state_spec, self.terminal_spec,
             self.weight_spec),
            initializers=(
                tf.zeros, tf.zeros, tf.zeros, tf.ones))

        # Meta information.
        self.output_specs = Transition(state=self.state_spec,
                                       reward=self.reward_spec,
                                       next_state=self.next_state_spec,
                                       terminal=self.terminal_spec,
                                       weight=self.weight_spec)
        self.output_dtypes = tf.nest.map_structure(
            lambda spec: spec.dtype, self.output_specs)
        self.output_shapes = tf.nest.map_structure(
            lambda spec: spec.shape, self.output_specs)

    def __getattr__(self, name):
        return getattr(self._env, name)

    def _py_reset(self):
        next_state = self._env.reset()
        return next_state

    def _py_step(self, action):
        next_state, reward, terminal, _ = self._env.step(action.numpy())
        return next_state, reward, terminal

    def _py_seed(self, seed):
        self._env.seed(seed.numpy().item())

    def _py_close(self):
        self._env.close()

    def seed(self, seed):
        tf.py_function(self._py_seed, (seed,), ())

    def close(self):
        tf.py_function(self._py_close, (), ())

    def _fit_shape(self, spec):
        shape = spec.shape.as_list()
        def shape_fn(tensor):
            if not self._is_batched:
                tensor = tf.expand_dims(tensor, axis=0)
            tensor.set_shape(
                self._output_shape_list + shape)
            return tensor
        return shape_fn

    def reset(self):
        next_state = tf.py_function(
            self._py_reset, (),
            self.output_dtypes.next_state)
        next_state = tf.nest.map_structure(
            lambda s, spec: self._fit_shape(spec)(s),
            next_state,
            self.output_specs.next_state)
        return Transition(state=self._initial_state,
                          reward=self._initial_reward,
                          next_state=next_state,
                          terminal=self._initial_terminal,
                          weight=self._initial_weight)

    def step(self, agent_outputs, env_outputs, time_step):
        # Force the shape of the actions to match the environment.
        action = agent_outputs.action
        if not self._is_batched:
            action = action[0]

        next_state, reward, terminal = tf.py_function(
            self._py_step, (action,),
            (self.output_dtypes.next_state,
             self.output_dtypes.reward,
             self.output_dtypes.terminal))

        next_state = tf.nest.map_structure(
            lambda s, spec: self._fit_shape(spec)(s), next_state,
            self.output_specs.next_state)
        reward = tf.nest.map_structure(
            lambda s, spec: self._fit_shape(spec)(s), reward,
            self.output_specs.reward)
        terminal = tf.nest.map_structure(
            lambda s, spec: self._fit_shape(spec)(s), terminal,
            self.output_specs.terminal)
        terminal = tf.logical_or(terminal, env_outputs.terminal)
        weight = tf.cast(
            tf.logical_not(env_outputs.terminal),
            tf.dtypes.float32)

        return Transition(state=env_outputs.next_state,
                          reward=reward,
                          next_state=next_state,
                          terminal=terminal,
                          weight=weight)
