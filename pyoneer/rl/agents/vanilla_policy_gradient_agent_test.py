from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.platform import test
from tensorflow.python.framework import test_util

from pyoneer.rl.agents import vanilla_policy_gradient_agent_impl
from pyoneer.rl.agents.test import policy_value_test_case_impl


class DiscreteVanillaPolicyGradientAgentTest(policy_value_test_case_impl.PolicyValueTestCase):

    @test_util.skip_if(True)
    def testConvergenceDiscrete(self):
        with context.eager_mode():
            self.setUpEnv(
                'CartPole-v0',
                seed=42,
                clip_inf=40.)
            self.setUpDiscretePolicy()
            self.setUpLinearStateValue()
            self.setUpOptimizer()
            agent = vanilla_policy_gradient_agent_impl.VanillaPolicyGradientAgent(
                policy=self.policy, 
                value=self.value, 
                optimizer=self.optimizer)
            self.assertNaiveStrategyConvergedAfter(
                agent,
                iterations=150,
                explore_episodes=128,
                explore_max_steps=200,
                exploit_episodes=10,
                exploit_max_steps=200,
                max_returns=200.)

    @test_util.skip_if(True)
    def testConvergenceRecurrentDiscrete(self):
        with context.eager_mode():
            self.setUpEnv(
                'CartPole-v0',
                seed=42,
                clip_inf=40.)
            self.setUpRecurrentDiscretePolicy()
            self.setUpLinearStateValue()
            self.setUpOptimizer()
            agent = vanilla_policy_gradient_agent_impl.VanillaPolicyGradientAgent(
                policy=self.policy, 
                value=self.value, 
                optimizer=self.optimizer)
            self.assertNaiveStrategyConvergedAfter(
                agent,
                iterations=150,
                explore_episodes=128,
                explore_max_steps=200,
                exploit_episodes=10,
                exploit_max_steps=200,
                max_returns=200.)

    @test_util.skip_if(True)
    def testConvergenceContinuous(self):
        with context.eager_mode():
            self.setUpEnv(
                'Pendulum-v0',
                seed=42,
                clip_inf=40.)
            self.setUpContinuousPolicy(scale=2.)
            self.setUpLinearStateValue()
            self.setUpOptimizer()
            agent = vanilla_policy_gradient_agent_impl.VanillaPolicyGradientAgent(
                policy=self.policy, 
                value=self.value, 
                optimizer=self.optimizer)
            self.assertNaiveStrategyConvergedAfter(
                agent,
                iterations=150,
                explore_episodes=128,
                explore_max_steps=250,
                exploit_episodes=10,
                exploit_max_steps=250,
                max_returns=-200.)

    @test_util.skip_if(True)
    def testConvergenceRecurrentContinuous(self):
        with context.eager_mode():
            self.setUpEnv(
                'Pendulum-v0',
                seed=42,
                clip_inf=40.)
            self.setUpRecurrentContinuousPolicy(scale=2.)
            self.setUpLinearStateValue()
            self.setUpOptimizer()
            agent = vanilla_policy_gradient_agent_impl.VanillaPolicyGradientAgent(
                policy=self.policy, 
                value=self.value, 
                optimizer=self.optimizer)
            self.assertNaiveStrategyConvergedAfter(
                agent,
                iterations=150,
                explore_episodes=128,
                explore_max_steps=250,
                exploit_episodes=10,
                exploit_max_steps=250,
                max_returns=-200.)


if __name__ == "__main__":
    test.main()