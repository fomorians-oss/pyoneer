from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import control_flow_ops

from pyoneer.rl import rollout_impl


def parallel_rollout(env,
                     next_action_fn,
                     initial_state=None,
                     initial_action=None,
                     initial_reward=None,
                     initial_done=None,
                     episodes=1,
                     max_steps=100,
                     done_on_max_steps=False):
    state_dtype = dtypes.as_dtype(env.observation_space.dtype)
    action_dtype = dtypes.as_dtype(env.action_space.dtype)

    states_ta = tensor_array_ops.TensorArray(
        state_dtype, size=max_steps, name='states_ta')
    actions_ta = tensor_array_ops.TensorArray(
        action_dtype, size=max_steps, name='actions_ta')
    rewards_ta = tensor_array_ops.TensorArray(
        dtypes.float32, size=max_steps, name='rewards_ta')
    dones_ta = tensor_array_ops.TensorArray(
        dtypes.bool, size=max_steps, name='dones_ta')

    def condition(state, state_ta,
                  action, action_ta,
                  reward, reward_ta,
                  done, done_ta, i):
        step_limit = gen_math_ops.less(i, max_steps)
        done_limit = math_ops.reduce_sum(
            math_ops.cast(done, dtypes.int32)) != array_ops.shape(done)[0]
        return step_limit and done_limit

    def body(state, state_ta,
             action, action_ta,
             reward, reward_ta,
             done, done_ta, i):
        state = math_ops.cast(state, state_dtype)
        state_ta = state_ta.write(i, state)
        action = next_action_fn(i, state, action, reward, done, False)
        next_state, reward, next_done, _ = env.step(action)
        done = gen_math_ops.logical_or(done, next_done)
        action_ta = action_ta.write(i, math_ops.cast(action, action_dtype))
        reward_ta = reward_ta.write(i, math_ops.cast(reward, dtypes.float32))
        done_ta = done_ta.write(i, done)
        return (
            next_state, state_ta,
            action, action_ta,
            reward, reward_ta,
            done, done_ta,
            gen_math_ops.add(i, 1))

    if isinstance(initial_state, ops.Tensor):
        state = array_ops.identity(initial_state)
    else:
        state = env.reset(episodes)

    state_shape = state.shape[1:]
    state = math_ops.cast(state, state_dtype)

    action = next_action_fn(0, state, initial_action, initial_reward, initial_done, True)
    action_shape = action.shape[1:]
    next_state, reward, done, _ = env.step(action)

    states_ta = states_ta.write(0, state)
    actions_ta = actions_ta.write(0, math_ops.cast(action, action_dtype))
    rewards_ta = rewards_ta.write(0, math_ops.cast(reward, dtypes.float32))
    dones_ta = dones_ta.write(0, ops.convert_to_tensor(done))

    idx = array_ops.constant(1)
    _, states_ta, _, actions_ta, _, rewards_ta, _, dones_ta, _ = control_flow_ops.while_loop(
        condition, body, [
            next_state, states_ta,
            action, actions_ta,
            reward, rewards_ta,
            done, dones_ta,
            idx])

    if done_on_max_steps:
        dones_ta = dones_ta.write(max_steps - 1, array_ops.ones_like(done))

    states = states_ta.stack()
    states = array_ops.transpose(states, [1, 0] + list(range(2, 2 + len(state_shape))))
    actions = actions_ta.stack()
    actions = array_ops.transpose(actions, [1, 0] + list(range(2, 2 + len(action_shape))))
    rewards = rewards_ta.stack()
    rewards = array_ops.transpose(rewards, [1, 0])
    dones = dones_ta.stack()
    dones = array_ops.transpose(dones, [1, 0])

    return rollout_impl.ContiguousRollout(
        states=states, 
        actions=actions, 
        rewards=rewards, 
        weights=1. - math_ops.cast(dones, dtypes.float32))
