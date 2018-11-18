import argparse
import os
from gym import logger
import random
import numpy as np

import tensorflow as tf
import tensorflow.contrib.eager as tfe

from pyoneer.core import batch_gym
from pyoneer.core.normalizers import high_low

from pyoneer.baselines.policies import continuous_control
from pyoneer.baselines.values import linear_baseline
from pyoneer.baselines.experiments import vpg_experiment


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
    action_normalizer,
    layers=[400, 300],
    init_stdev=4.)
baseline = linear_baseline.LinearBaseline()
global_step = tfe.Variable(0, trainable=False)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)

vpg_experiment.run_vpg_experiment(
    policy, 
    baseline,
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