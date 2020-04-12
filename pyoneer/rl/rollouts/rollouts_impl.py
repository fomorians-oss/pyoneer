import numpy as np
import tensorflow as tf

from collections import defaultdict


class Rollout:
    """
    Nest-compatible API for arbitrarily nested transitions.
    Allows environments to act as "transition producers",
    while rollouts act as "transition accumulators".
    """

    def __init__(self, env):
        self.env = env

    def __call__(self, policy, episodes, render_mode=None, include_info=False):
        observation_space = self.env.observation_space
        action_space = self.env.action_space
        max_episode_steps = self.env.spec.max_episode_steps

        batch_size = min(len(self.env), episodes)
        batches = episodes // batch_size

        observations = np.zeros(
            shape=(episodes, max_episode_steps) + observation_space.shape,
            dtype=observation_space.dtype,
        )
        actions = np.zeros(
            shape=(episodes, max_episode_steps) + action_space.shape,
            dtype=action_space.dtype,
        )
        observations_next = np.zeros(
            shape=(episodes, max_episode_steps) + observation_space.shape,
            dtype=observation_space.dtype,
        )
        rewards = np.zeros(shape=(episodes, max_episode_steps), dtype=np.float32)
        weights = np.zeros(shape=(episodes, max_episode_steps), dtype=np.float32)
        dones = np.ones(shape=(episodes, max_episode_steps), dtype=np.bool)

        if render_mode == "rgb_array":
            renders = []

        if include_info:
            infos = defaultdict(list)

        for batch in range(batches):
            batch_start = batch * batch_size
            batch_end = batch_start + batch_size

            episode_done = np.zeros(shape=batch_size, dtype=np.bool)
            observation = self.env.reset()

            for step in range(max_episode_steps):
                if render_mode == "rgb_array":
                    renders.append(self.env.render(mode="rgb_array"))
                elif render_mode is not None:
                    self.env.render(mode=render_mode)

                reset_state = step == 0

                observation = observation.astype(observation_space.dtype)
                actions_batch = policy(
                    observation[:batch_size, None],
                    training=False,
                    reset_state=reset_state,
                )
                action = actions_batch[:, 0].numpy()
                action = action.astype(action_space.dtype)

                observation_next, reward, done, info = self.env.step(action)

                observations[batch_start:batch_end, step] = observation[:batch_size]
                actions[batch_start:batch_end, step] = action[:batch_size]
                observations_next[batch_start:batch_end, step] = observation_next[
                    :batch_size
                ]
                rewards[batch_start:batch_end, step] = reward[:batch_size]
                weights[batch_start:batch_end, step] = np.where(
                    episode_done[:batch_size], 0.0, 1.0
                )
                dones[batch_start:batch_end, step] = done[:batch_size]

                if include_info:
                    for key, val in info.items():
                        infos[key].append(val)

                episode_done = episode_done | done[:batch_size]

                # end the rollout if all episodes are done
                if np.all(episode_done):
                    break

                observation = observation_next

        transitions = {
            "observations": observations,
            "actions": actions,
            "observations_next": observations_next,
            "rewards": rewards,
            "weights": weights,
            "dones": dones,
        }

        if render_mode == "rgb_array":
            transitions["renders"] = np.stack(renders, axis=0)

        if include_info:
            transitions["infos"] = {}
            for key, val in infos.items():
                if len(val) > 0:
                    transitions["infos"][key] = np.stack(val, axis=1)

        return transitions
