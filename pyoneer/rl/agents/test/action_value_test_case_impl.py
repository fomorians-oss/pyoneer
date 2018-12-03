from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np

from gym import logger
from gym.spaces import box
from gym.spaces import discrete

import tensorflow as tf
import tensorflow.contrib.eager as tfe
from tensorflow.python.platform import test
from tensorflow.python.eager import context

from pyoneer.rl import batch_gym
from pyoneer.rl import parallel_rollout_impl
from pyoneer.rl import rollout_impl
from pyoneer.rl.agents.test import gym_test_utils
from pyoneer.rl.agents.test import naive_strategy_impl
from pyoneer.rl.agents.test.policy import discrete_policy_impl
from pyoneer.rl.agents.test.policy import continuous_policy_impl
from pyoneer.rl.agents.test.value import linear_value_impl
from pyoneer.rl.agents.test.value import non_linear_value_impl


class ActionValueTestCase(test.TestCase):
    """Policy-Value test case for discrete and continuous spaces.
    """

    def setUp(self):
        super(ActionValueTestCase, self).setUp()
        logger.setLevel(logger.DISABLED)

    def setUpEnv(self, 
                 env_spec,
                 seed=42,
                 clip_inf=40.):
        random.seed(seed)
        np.random.seed(seed)
        tf.set_random_seed(seed)

        self.explore_env = batch_gym.batch_gym_make(env_spec)
        self.assertTrue(isinstance(self.explore_env.observation_space, box.Box))
        self.explore_env.seed(seed)
        self.exploit_env = batch_gym.batch_gym_make(env_spec)
        self.exploit_env.seed(seed)

        self.state_normalizer = gym_test_utils.high_low_normalizer_from_gym_space(
            self.explore_env.observation_space, clip_inf=clip_inf)

    def setUpNonLinearStateValue(self):
        self.assertTrue(isinstance(self.explore_env.action_space, discrete.Discrete))
        self.value = non_linear_value_impl.NonLinearStateValue(
            self.state_normalizer, units=self.explore_env.action_space.n)

    def setUpTargetNonLinearStateValue(self):
        self.assertTrue(isinstance(self.explore_env.action_space, discrete.Discrete))
        self.target_value = non_linear_value_impl.NonLinearStateValue(
            self.state_normalizer, units=self.explore_env.action_space.n)

    def setUpOptimizer(self, learning_rate=1e-3):
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    # TODO(wenkesj): make this work. rollouts with intermediate callbacks for training.
    def assertNaiveStrategyConvergedAfter(self, 
                                          agent,
                                          iterations=100, 
                                          epochs=1,
                                          batch_size=128,
                                          explore_episodes=128,
                                          explore_max_steps=200,
                                          exploit_episodes=10,
                                          exploit_max_steps=200,
                                          max_returns=200.,
                                          after_iteration=lambda agent: None,
                                          contiguous=True):
        self.strategy = naive_strategy_impl.NaiveValueStrategy(self.value)
        explore_rollouts = parallel_rollout_impl.parallel_rollout(
            self.explore_env,
            self.strategy.explore,
            episodes=explore_episodes,
            max_steps=explore_max_steps,
            contiguous=contiguous)

        for _ in range(iterations):
            explore_rollouts = parallel_rollout_impl.parallel_rollout(
                self.explore_env,
                self.strategy.explore,
                episodes=explore_episodes,
                max_steps=explore_max_steps,
                contiguous=contiguous).concat(explore_rollouts, axis=0)

            dataset = tf.data.Dataset.from_tensor_slices(explore_rollouts)
            dataset = dataset.batch(batch_size)
            dataset = dataset.shuffle(4)

            for _ in range(epochs):
                for batch_rollouts in tfe.Iterator(dataset):
                    agent.fit(batch_rollouts)
                    exploit_rollouts = parallel_rollout_impl.parallel_rollout(
                        self.exploit_env,
                        self.strategy.exploit,
                        episodes=exploit_episodes,
                        max_steps=exploit_max_steps)
                    mean_episodic_exploit_returns = tf.reduce_mean(
                        tf.reduce_sum(exploit_rollouts.rewards, axis=-1))
                    print(mean_episodic_exploit_returns)
                    if mean_episodic_exploit_returns.numpy() >= max_returns:
                        self.assertAllGreaterEqual(mean_episodic_exploit_returns, tf.constant(max_returns))
                        return
            after_iteration(agent)

        self.assertAllGreaterEqual(mean_episodic_exploit_returns, tf.constant(max_returns))
