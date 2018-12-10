import numpy as np
from gym import logger; logger.set_level(logger.DISABLED)
import tensorflow as tf; tf.enable_eager_execution()
import tensorflow.contrib.eager as tfe
import tensorflow_probability as tfp
import pyoneer as pynr
import pyoneer.rl as pyrl
import pyoneer.contrib.interactor as pyactor

class Policy(tf.keras.Model):
    def __init__(self, 
                 state_normalizer, 
                 action_normalizer, 
                 reward_normalizer, 
                 action_size):
        super(Policy, self).__init__()
        self.state_normalizer = state_normalizer
        self.action_normalizer = action_normalizer
        self.reward_normalizer = reward_normalizer
        self.hidden = tf.layers.Dense(
            64, 
            activation=pynr.nn.swish, 
            kernel_initializer=tf.initializers.variance_scaling(scale=2.0))
        self.rnn = pynr.layers.RNN(tf.nn.rnn_cell.LSTMCell(64))
        self.outputs = tf.layers.Dense(
            action_size,
            kernel_initializer=tf.initializers.variance_scaling(scale=2.0))

    def call(self, inputs, reset_state=True, **kwargs):
        states, actions, rewards = inputs
        states_norm = self.state_normalizer(states)
        actions_norm = self.action_normalizer(actions)
        rewards_norm = self.reward_normalizer(tf.expand_dims(rewards, axis=-1))
        hidden = self.hidden(states_norm)
        hidden = tf.concat([states_norm, actions_norm, rewards_norm], axis=-1)
        hidden = self.rnn(hidden, reset_state=reset_state)
        outputs = self.outputs(hidden)
        return tfp.distributions.Categorical(logits=outputs)

class Value(tf.keras.Model):

    def __init__(self,
                 state_normalizer, 
                 action_normalizer, 
                 reward_normalizer):
        super(Value, self).__init__()
        self.state_normalizer = state_normalizer
        self.action_normalizer = action_normalizer
        self.reward_normalizer = reward_normalizer
        kernel_initializer = tf.initializers.variance_scaling(scale=2.0)
        self.hidden = tf.layers.Dense(
            64, 
            activation=pynr.nn.swish,
            kernel_initializer=kernel_initializer)
        self.rnn = pynr.layers.RNN(tf.nn.rnn_cell.LSTMCell(64))
        self.value = tf.layers.Dense(
            1, 
            kernel_initializer=kernel_initializer)

    def call(self, inputs, reset_state=True, **kwargs):
        states, actions, rewards = inputs
        states_norm = self.state_normalizer(states)
        actions_norm = self.action_normalizer(actions)
        rewards_norm = self.reward_normalizer(tf.expand_dims(rewards, axis=-1))
        hidden = self.hidden(states_norm)
        hidden = tf.concat([states_norm, actions_norm, rewards_norm], axis=-1)
        hidden = self.rnn(hidden, reset_state=reset_state)
        return self.value(hidden)

def actor(strategy, reward_normalizer, deterministic=False):
    def actor_fn(i, state, action, reward, done, is_initial_state):
        if deterministic:
            actions = strategy.policy(
                (state, action, reward), 
                reset_state=is_initial_state).mode()
        else:
            _ = reward_normalizer(tf.expand_dims(reward, axis=-1), training=True)
            actions = strategy(
                (state, action, reward), 
                reset_state=is_initial_state)
        return tf.cast(actions, tf.int32)
    return actor_fn

num_iterations = 100
num_explore_episodes = 256
num_explore_max_steps = 200
num_exploit_episodes = 10
num_exploit_max_steps = 200
returns_threshold = 199.

explore_env = pyactor.batch_gym_make('CartPole-v0')
exploit_env = pyactor.batch_gym_make('CartPole-v0')

state_normalizer = pynr.features.HighLowNormalizer(
    np.clip(explore_env.observation_space.high, -20., 20.),
    np.clip(explore_env.observation_space.low, -20., 20.), 
    dtype=tf.float32)
action_normalizer = pynr.layers.OneHotEncoder(explore_env.action_space.n)
reward_normalizer = pynr.features.SampleAverageNormalizer([1])

policy = Policy(
    state_normalizer, 
    action_normalizer,
    reward_normalizer,
    explore_env.action_space.n)
value = Value(
    state_normalizer, 
    action_normalizer,
    reward_normalizer)

global_step = tfe.Variable(0, dtype=tf.int64)
epsilon = tf.train.exponential_decay(
    .5, global_step, num_iterations, .99)
strategy = pyrl.strategies.EpsilonGreedyStrategy(policy, epsilon)

agent = pyrl.agents.AdvantageActorCriticAgent(
    policy=policy, 
    value=value,
    optimizer=tf.train.AdamOptimizer(1e-3))

for _ in range(num_iterations):
    explore_rollouts = pyactor.batch_rollout(
        explore_env,
        actor(strategy, reward_normalizer),
        initial_action=tf.zeros([num_explore_episodes], tf.int32),
        initial_reward=tf.zeros([num_explore_episodes], tf.float32),
        episodes=num_explore_episodes,
        max_steps=num_explore_max_steps)
    _ = agent.fit(
        states=(
            explore_rollouts.states,
            explore_rollouts.actions,
            explore_rollouts.rewards),
        actions=explore_rollouts.actions,
        rewards=explore_rollouts.rewards,
        weights=explore_rollouts.weights,
        global_step=global_step)
    exploit_rollouts = pyactor.batch_rollout(
        exploit_env,
        actor(strategy, reward_normalizer, deterministic=True),
        initial_action=tf.zeros([num_exploit_episodes], tf.int32),
        initial_reward=tf.zeros([num_exploit_episodes], tf.float32),
        episodes=num_exploit_episodes,
        max_steps=num_exploit_max_steps)
    mean_episodic_exploit_returns = tf.reduce_mean(
        tf.reduce_sum(exploit_rollouts.rewards, axis=-1))
    print(mean_episodic_exploit_returns)
    if mean_episodic_exploit_returns.numpy() > returns_threshold:
        break
