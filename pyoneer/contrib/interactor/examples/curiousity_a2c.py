import numpy as np
from gym import logger; logger.set_level(logger.DISABLED)
import tensorflow as tf; tf.enable_eager_execution()
import tensorflow.contrib.eager as tfe
import tensorflow_probability as tfp
import pyoneer as pynr
import pyoneer.rl as pyrl
import pyoneer.contrib.interactor as pyactor

class Policy(tf.keras.Model):
    def __init__(self, state_normalizer, action_size):
        super(Policy, self).__init__()
        self.state_normalizer = state_normalizer
        self.hidden = tf.layers.Dense(
            64, 
            activation=pynr.nn.swish, 
            kernel_initializer=tf.initializers.variance_scaling(scale=2.0))
        self.outputs = tf.layers.Dense(
            action_size,
            kernel_initializer=tf.initializers.variance_scaling(scale=2.0))

    def call(self, inputs, **kwargs):
        norm = self.state_normalizer(inputs)
        hidden = self.hidden(norm)
        outputs = self.outputs(hidden)
        return tfp.distributions.Categorical(logits=outputs)

class Value(tf.keras.Model):

    def __init__(self, state_normalizer):
        super(Value, self).__init__()
        self.state_normalizer = state_normalizer
        kernel_initializer = tf.initializers.variance_scaling(scale=2.0)
        self.hidden = tf.layers.Dense(
            64, 
            activation=pynr.nn.swish,
            kernel_initializer=kernel_initializer)
        self.value = tf.layers.Dense(
            1, 
            kernel_initializer=kernel_initializer)

    def call(self, states, **kwargs):
        states_norm = self.state_normalizer(states)
        hidden = self.hidden(states_norm)
        return self.value(hidden)

class Model(tf.keras.Model):

    def __init__(self, state_normalizer, num_units, action_size):
        super(Model, self).__init__()
        self.state_normalizer = state_normalizer
        self.action_size = action_size
        kernel_initializer = tf.initializers.variance_scaling(scale=2.0)
        self.hidden = tf.layers.Dense(
            64, 
            activation=pynr.nn.swish,
            kernel_initializer=kernel_initializer)
        self.outputs = tf.layers.Dense(
            num_units, 
            kernel_initializer=kernel_initializer)

    def call(self, states, actions, **kwargs):
        states_norm = self.state_normalizer(states)
        actions_norm = tf.one_hot(actions, self.action_size, dtype=tf.float32)
        hidden = self.hidden(tf.concat([states_norm, actions_norm], axis=-1))
        return self.outputs(hidden)

def actor(strategy, deterministic=False):
    def actor_fn(i, state, action, reward, done, is_initial_state):
        if deterministic:
            actions = strategy.policy(state).mode()
        else:
            actions = strategy(state)
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
reward_normalizer = pynr.features.SampleAverageNormalizer([1], center=False)
policy = Policy(state_normalizer, explore_env.action_space.n)
value = Value(state_normalizer)
forward_model = Model(
    state_normalizer, explore_env.observation_space.shape[-1], explore_env.action_space.n)
global_step = tfe.Variable(0, dtype=tf.int64)
epsilon = tf.train.exponential_decay(
    .5, global_step, num_iterations, .99)
strategy = pyrl.strategies.EpsilonGreedyStrategy(policy, epsilon)
agent = pyrl.agents.IntrinsicMotivationAdvantageActorCriticAgent(
    policy=policy, 
    value=value,
    forward_model=forward_model,
    optimizer=tf.train.AdamOptimizer(1e-3),
    forward_optimizer=tf.train.AdamOptimizer(1e-3))
transformation = pyrl.transformations.IntrinsicMotivationBonusTransformation(
    forward_model=forward_model)

for _ in range(num_iterations):
    explore_rollouts = pyactor.batch_rollout(
        explore_env,
        actor(strategy),
        episodes=num_explore_episodes,
        max_steps=num_explore_max_steps)
    transformed_rewards = transformation(
        explore_rollouts.states, 
        explore_rollouts.next_states, 
        explore_rollouts.actions, 
        explore_rollouts.rewards, 
        explore_rollouts.weights, 
        bonus_scale=1.)

    intrinsic_rewards, extrinsic_rewards = tf.unstack(transformed_rewards, axis=-1)
    intrinsic_rewards = tf.squeeze(
        reward_normalizer(tf.expand_dims(intrinsic_rewards, axis=-1), training=True), 
        axis=-1)
    transformed_rewards = intrinsic_rewards + extrinsic_rewards

    _ = agent.fit(
        states=explore_rollouts.states,
        next_states=explore_rollouts.next_states,
        actions=explore_rollouts.actions,
        rewards=transformed_rewards,
        weights=explore_rollouts.weights,
        global_step=global_step)

    exploit_rollouts = pyactor.batch_rollout(
        exploit_env,
        actor(strategy, deterministic=True),
        episodes=num_exploit_episodes,
        max_steps=num_exploit_max_steps)
    mean_episodic_exploit_returns = tf.reduce_mean(
        tf.reduce_sum(exploit_rollouts.rewards, axis=-1))
    print(mean_episodic_exploit_returns)
    if mean_episodic_exploit_returns.numpy() > returns_threshold:
        break
