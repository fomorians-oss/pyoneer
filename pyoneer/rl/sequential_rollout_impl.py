from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

from pyoneer.rl import rollout_impl


def sequential_rollout(env,
                       next_action_fn,
                       initial_state=None,
                       initial_action=None,
                       initial_reward=None,
                       initial_done=None,
                       episodes=1,
                       max_steps=100,
                       done_on_max_steps=False):
    max_steps_fn = lambda s: s >= max_steps

    def episode_fn(env):
        """Run an episode in the environment."""
        step = 0
        inner_states = []
        inner_actions = []
        inner_rewards = []
        inner_dones = []

        state = env.reset()
        inner_states.append(state)
        action = next_action_fn(0, state, initial_action, initial_reward, initial_done, True)
        inner_actions.append(action)

        while True:
            state, reward, done, _ = env.step(action)
            inner_dones.append(done)
            inner_rewards.append(reward)
            step += 1
            if max_steps_fn(step) or done:
                break
            inner_states.append(state)
            action = next_action_fn(step, state, action, reward, done, False)
            inner_actions.append(action)
        
        if done_on_max_steps:
            inner_dones[-1] = True

        return rollout_impl.ContiguousRollout(
            states=array_ops.expand_dims(
                array_ops.stack(inner_states, axis=0), axis=0), 
            actions=array_ops.expand_dims(
                array_ops.stack(inner_actions, axis=0), axis=0),
            rewards=array_ops.expand_dims(
                array_ops.stack(inner_rewards, axis=0), axis=0),
            weights=array_ops.expand_dims(
                array_ops.stack(1. - math_ops.cast(inner_dones, dtypes.float32), axis=0), axis=0))

    rollouts = episode_fn(env)
    for _ in range(episodes - 1):
        rollouts = rollouts.concat(episode_fn(env), axis=0)

    return rollouts
