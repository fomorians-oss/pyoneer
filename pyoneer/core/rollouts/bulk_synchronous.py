import tensorflow as tf

from pyoneer.core.rollouts import rollout


def bulk_synchronous_rollout(env,
                             next_action_fn,
                             initial_state=None,
                             initial_action=None,
                             initial_reward=None,
                             initial_done=None,
                             episodes=1,
                             max_steps=100,
                             done_on_max_steps=False):
    """Rollout an env.

    Args:
      env: instance of `Env`
      next_action_fn:
        callable that has the signature `next_action_fn(step, state, action, reward, done, is_initial_state)`.

        Example:
          ```
          def next_action_fn(step, state, action, reward, done, is_initial_state):
            action = agent(
              tf.expand_dims(state, axis=1),
              training=False,
              reset_state=is_initial_state)
            return tf.squeeze(action, axis=1)
          ```

      initial_action: action to be passed to in first `next_action_fn` call.
      initial_reward: reward to be passed to in first `next_action_fn` call.
      initial_done: done to be passed to in first `next_action_fn` call.
      episodes: number of episodes to be rolled.
      max_steps: maximum number of steps per episode.
      done_on_max_steps: for `done = True` when `max_steps` is reached.

    Returns:
      (states, actions, rewards, weights)
    """
    state_dtype = tf.as_dtype(env.observation_space.dtype)
    action_dtype = tf.as_dtype(env.action_space.dtype)

    states_ta = tf.TensorArray(
        state_dtype, size=max_steps, name='states_ta')
    actions_ta = tf.TensorArray(
        action_dtype, size=max_steps, name='actions_ta')
    rewards_ta = tf.TensorArray(
        tf.float32, size=max_steps, name='rewards_ta')
    dones_ta = tf.TensorArray(
        tf.bool, size=max_steps, name='dones_ta')

    def condition(state, state_ta,
                  action, action_ta,
                  reward, reward_ta,
                  done, done_ta, i):
        step_limit = tf.less(i, max_steps)
        done_limit = tf.reduce_sum(tf.cast(done, tf.int32)) != tf.shape(done)[0]
        return step_limit and done_limit

    def body(state, state_ta,
             action, action_ta,
             reward, reward_ta,
             done, done_ta, i):
        state = tf.cast(state, state_dtype)
        state_ta = state_ta.write(i, state)
        action = next_action_fn(i, state, action, reward, done, False)
        next_state, reward, next_done, _ = env.step(action)
        done = tf.logical_or(done, next_done)
        action_ta = action_ta.write(i, tf.cast(action, action_dtype))
        reward_ta = reward_ta.write(i, tf.cast(reward, tf.float32))
        done_ta = done_ta.write(i, done)
        return (
            next_state, state_ta,
            action, action_ta,
            reward, reward_ta,
            done, done_ta,
            tf.add(i, 1))

    if isinstance(initial_state, tf.Tensor):
        state = tf.identity(initial_state)
    else:
        state = env.reset(episodes)

    state_shape = state.shape[1:]
    state = tf.cast(state, state_dtype)

    action = next_action_fn(0, state, initial_action, initial_reward, initial_done, True)
    action_shape = action.shape[1:]
    next_state, reward, done, _ = env.step(action)

    states_ta = states_ta.write(0, state)
    actions_ta = actions_ta.write(0, tf.cast(action, action_dtype))
    rewards_ta = rewards_ta.write(0, tf.cast(reward, tf.float32))
    dones_ta = dones_ta.write(0, tf.convert_to_tensor(done))

    idx = tf.constant(1)
    _, states_ta, _, actions_ta, _, rewards_ta, _, dones_ta, _ = tf.while_loop(
        condition, body, [
            next_state, states_ta,
            action, actions_ta,
            reward, rewards_ta,
            done, dones_ta,
            idx])

    if done_on_max_steps:
        dones_ta = dones_ta.write(max_steps - 1, tf.ones_like(done))

    states = states_ta.stack()
    states = tf.transpose(states, [1, 0] + list(range(2, 2 + len(state_shape))))
    actions = actions_ta.stack()
    actions = tf.transpose(actions, [1, 0] + list(range(2, 2 + len(action_shape))))
    rewards = rewards_ta.stack()
    rewards = tf.transpose(rewards, [1, 0])
    dones = dones_ta.stack()
    dones = tf.transpose(dones, [1, 0])

    return rollout.Rollout(states=states, 
                           actions=actions, 
                           rewards=rewards, 
                           weights=1. - tf.cast(dones, tf.float32))
