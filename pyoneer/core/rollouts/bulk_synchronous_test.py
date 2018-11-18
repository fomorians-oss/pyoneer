import tensorflow as tf

from pyoneer.core.rollouts import bulk_synchronous


class MockSpace(object):
    
    def __init__(self, dtype):
        self.dtype = dtype


class CounterEnv(object):

    def __init__(self):
        self.observation_space = MockSpace(tf.float32)
        self.action_space = MockSpace(tf.float32)
        self._step = 0

    def reset(self, size):
        self._step = 1
        return tf.zeros(size, dtype=self.observation_space.dtype)

    def step(self, action):
        zero = tf.zeros_like(action)
        state = tf.cast(zero, self.observation_space.dtype) + self._step
        self._step += 1
        return state, tf.cast(zero, tf.float32), tf.cast(zero, tf.bool), {}


class BulkSynchronousTest(tf.test.TestCase):

    def test_bulk_synchronous_rollout(self):
        test_env = CounterEnv()

        def next_action_fn(i, state, action, reward, done, is_initial_state):
            return tf.zeros_like(state)

        episodes = 5
        max_steps = 100
        rollout = bulk_synchronous.bulk_synchronous_rollout(
            test_env,
            next_action_fn,
            initial_state=None,
            initial_action=None,
            initial_reward=None,
            initial_done=None,
            episodes=episodes,
            max_steps=max_steps,
            done_on_max_steps=False)

        expected_states = tf.tile(tf.expand_dims(tf.range(max_steps), 0), [episodes, 1])
        expected_actions = tf.zeros_like(expected_states)
        expected_rewards = tf.zeros_like(expected_states)
        expected_weights = tf.ones_like(expected_states)

        self.assertAllClose(rollout.states, expected_states, atol=1e-8)
        self.assertAllClose(rollout.actions, expected_actions, atol=1e-8)
        self.assertAllClose(rollout.rewards, expected_rewards, atol=1e-8)
        self.assertAllClose(rollout.weights, expected_weights, atol=1e-8)

    def test_bulk_synchronous_rollout_done_on_max_steps(self):
        test_env = CounterEnv()

        def next_action_fn(i, state, action, reward, done, is_initial_state):
            return tf.zeros_like(state)

        episodes = 5
        max_steps = 100
        rollout = bulk_synchronous.bulk_synchronous_rollout(
            test_env,
            next_action_fn,
            initial_state=None,
            initial_action=None,
            initial_reward=None,
            initial_done=None,
            episodes=episodes,
            max_steps=max_steps,
            done_on_max_steps=True)

        expected_states = tf.tile(tf.expand_dims(tf.range(max_steps), 0), [episodes, 1])
        expected_actions = tf.zeros_like(expected_states)
        expected_rewards = tf.zeros_like(expected_states)
        expected_weights = tf.concat(
            [tf.ones_like(expected_states[..., :-1]),
             tf.expand_dims(tf.zeros_like(expected_states[..., -1]), -1)], 
             axis=-1)

        self.assertAllClose(rollout.states, expected_states, atol=1e-8)
        self.assertAllClose(rollout.actions, expected_actions, atol=1e-8)
        self.assertAllClose(rollout.rewards, expected_rewards, atol=1e-8)
        self.assertAllClose(rollout.weights, expected_weights, atol=1e-8)


if __name__ == "__main__":
    tf.enable_eager_execution()
    tf.test.main()