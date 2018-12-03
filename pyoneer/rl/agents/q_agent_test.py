from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.platform import test
from tensorflow.python.framework import test_util

from pyoneer.rl.agents import q_agent_impl
from pyoneer.rl.agents.test import action_value_test_case_impl


class QAgentTest(action_value_test_case_impl.ActionValueTestCase):

    @test_util.skip_if(True)
    def testConvergenceDiscrete(self):
        with context.eager_mode():
            self.setUpEnv(
                'CartPole-v0',
                seed=42,
                clip_inf=40.)
            self.setUpNonLinearStateValue()
            self.setUpOptimizer()
            agent = q_agent_impl.QAgent(
                value=self.value, 
                optimizer=self.optimizer)
            self.assertNaiveStrategyConvergedAfter(
                agent,
                iterations=100,
                epochs=4,
                explore_episodes=128,
                explore_max_steps=200,
                exploit_episodes=10,
                exploit_max_steps=200,
                max_returns=200.,
                contiguous=False)


if __name__ == "__main__":
    test.main()