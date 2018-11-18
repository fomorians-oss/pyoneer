import os

import tensorflow as tf
import tensorflow.contrib.eager as tfe
import tensorflow_probability as tfp

from pyoneer.core import batch_gym
from pyoneer.core.rollouts import bulk_synchronous, rollout
from pyoneer.core.normalizers import high_low
from pyoneer.core.returns import discounted
from pyoneer.core.agents import a2c_agent

from pyoneer.baselines.experiments import utils


def run_a2c_experiment(policy, 
                       value,
                       optimizer,
                       global_step,
                       train_env, 
                       test_env,
                       job_dir,
                       iterations=30,
                       train_episodes=128, 
                       eval_episodes=10, 
                       max_steps=200, 
                       gamma=.999,
                       entropy_scale=.2):
    """
    Example:
        ```
        import argparse
        import os
        from gym import logger
        import random
        import numpy as np

        from pyoneer.baselines.policies import continuous_control
        from pyoneer.baselines.values import general_control

        parser = argparse.ArgumentParser()
        parser.add_argument('--job_dir', required=True)
        args = parser.parse_args()

        logger.set_level(logger.DISABLED)
        tf.enable_eager_execution()
        tf.logging.set_verbosity(tf.logging.INFO)

        random.seed(42)
        np.random.seed(42)
        tf.set_random_seed(42)

        train_env = batch_gym.batch_make('Pendulum-v0')
        test_env = batch_gym.batch_make('Pendulum-v0')

        state_normalizer = high_low.HighLowNormalizer(test_env.observation_space.low,
                                                      test_env.observation_space.high)
        action_normalizer = high_low.HighLowNormalizer(test_env.action_space.low,
                                                       test_env.action_space.high)

        policy = continuous_control.MultiVariateContinuousControlPolicy(
            state_normalizer, 
            action_normalizer)
        value = general_control.ValueFunction(state_normalizer)
        global_step = tfe.Variable(0, trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)

        run_a2c_experiment(
            policy, 
            value,
            optimizer,
            global_step,
            train_env, 
            test_env,
            args.job_dir,
            iterations=30,
            train_episodes=128, 
            eval_episodes=10, 
            max_steps=200, 
            gamma=.999,
            entropy_scale=.2)
        ```
    """
    agent = a2c_agent.A2CAgent(
        policy=policy,
        value=value,
        entropy_scale=.2)

    try:
        tf.gfile.MakeDirs(job_dir)
    except:
        pass

    summary_writer = tf.contrib.summary.create_file_writer(job_dir)
    summary_writer.set_as_default()

    checkpoint = tf.train.Checkpoint(agent=agent,
                                     global_step=global_step,
                                     optimizer=optimizer)
    checkpoint_path = tf.train.latest_checkpoint(job_dir)
    if checkpoint_path:
        tf.logging.info('Restoring {}...'.format(checkpoint_path))
        checkpoint.restore(checkpoint_path)

    tf.logging.info('Initializing agent.')
    state = tf.zeros([1, 1] + list(train_env.observation_space.shape),
                    dtype=tf.float32)
    agent.policy(state)
    agent.value(state)

    for iteration in range(iterations):
        tf.logging.info('Collecting exploration rollouts.')
        states, actions, rewards, weights = bulk_synchronous.bulk_synchronous_rollout(
            train_env,
            utils.sample_trajectories(agent, exploring=True),
            episodes=train_episodes,
            max_steps=max_steps)
        trajectories = rollout.Rollout(states, actions, rewards, weights)

        returns = discounted.discounted_returns(
            trajectories.rewards, 
            gamma, 
            weights=trajectories.weights)

        dataset = tf.data.Dataset.from_tensor_slices(
            (trajectories.states,
            trajectories.actions,
            tf.stop_gradient(returns),
            trajectories.weights)).batch(train_episodes)

        tf.logging.info('Training on rollouts.')
        for states, actions, returns, weights in tfe.Iterator(dataset):
            with tf.GradientTape() as tape:
                loss = sum(
                    agent.compute_loss(
                        states, actions, returns, weights, 
                        global_step=global_step, 
                        record_summaries=True))

            gradients = tape.gradient(loss, agent.trainable_variables)
            optimizer.apply_gradients(zip(gradients, agent.trainable_variables),
                                        global_step=global_step)

        tf.logging.info('Evaluating rollouts.')
        _, _, rewards, weights = bulk_synchronous.bulk_synchronous_rollout(
            test_env,
            utils.sample_trajectories(agent),
            episodes=eval_episodes,
            max_steps=max_steps)
        average_reward = tf.reduce_mean(tf.reduce_sum(rewards * weights, axis=-1))

        tf.logging.info(
            '{}/{} Reward = {}'.format(
                iteration + 1, iterations,
                average_reward.numpy()))
        with tf.contrib.summary.always_record_summaries():
            tf.contrib.summary.scalar('average_reward', average_reward, step=tf.cast(global_step, tf.int64))

        checkpoint_prefix = os.path.join(job_dir, 'ckpt')
        checkpoint.save(file_prefix=checkpoint_prefix)