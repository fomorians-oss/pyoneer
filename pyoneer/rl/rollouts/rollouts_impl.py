import numpy as np


class Rollout:
    """
    Nest-compatible API for arbitrarily nested transitions.
    Allows environments to act as "transition producers",
    while rollouts act as "transition accumulators".
    """

    def __init__(self, env, max_episode_steps):
        self.env = env
        self.max_episode_steps = max_episode_steps

    def __call__(self, policy, episodes, render=False):
        assert episodes % len(self.env) == 0

        observation_space = self.env.observation_space
        action_space = self.env.action_space

        batch_size = len(self.env)
        batches = episodes // batch_size

        observations = np.zeros(
            shape=(episodes, self.max_episode_steps) + observation_space.shape,
            dtype=observation_space.dtype,
        )
        actions = np.zeros(
            shape=(episodes, self.max_episode_steps) + action_space.shape,
            dtype=action_space.dtype,
        )
        observations_next = np.zeros(
            shape=(episodes, self.max_episode_steps) + observation_space.shape,
            dtype=observation_space.dtype,
        )
        rewards = np.zeros(shape=(episodes, self.max_episode_steps), dtype=np.float32)
        weights = np.zeros(shape=(episodes, self.max_episode_steps), dtype=np.float32)
        dones = np.ones(shape=(episodes, self.max_episode_steps), dtype=np.bool)

        for batch in range(batches):
            batch_start = batch * batch_size
            batch_end = batch_start + batch_size

            episode_done = np.zeros(shape=batch_size, dtype=np.bool)
            observation = self.env.reset()

            for step in range(self.max_episode_steps):
                if render:
                    self.env.render()

                reset_state = step == 0

                observation = observation.astype(observation_space.dtype)
                actions_batch = policy(
                    observation[:, None], training=False, reset_state=reset_state
                )
                action = actions_batch[:, 0].numpy()

                observation_next, reward, done, info = self.env.step(action)

                observations[batch_start:batch_end, step] = observation
                actions[batch_start:batch_end, step] = action
                observations_next[batch_start:batch_end, step] = observation_next
                rewards[batch_start:batch_end, step] = reward
                weights[batch_start:batch_end, step] = np.where(episode_done, 0.0, 1.0)
                dones[batch_start:batch_end, step] = done

                episode_done = episode_done | done

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

        return transitions
