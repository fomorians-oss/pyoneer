from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.platform import test
from tensorflow.python.framework import test_util

import trfl

from pyoneer.rl.agents import v_trace_proximal_policy_optimization_agent_impl
from pyoneer.rl.agents.test import policy_value_test_case_impl


class VTraceProximalPolicyOptimizationAgentTest(policy_value_test_case_impl.PolicyValueTestCase):

    @test_util.skip_if(True)
    def testConvergenceDiscrete(self):
        with context.eager_mode():
            self.setUpEnv(
                'CartPole-v0',
                seed=42,
                clip_inf=40.)
            self.setUpDiscretePolicy()
            self.setUpDiscreteBehavioralPolicy()
            self.setUpNonLinearStateValue()
            self.assignBehavioralPolicy()
            self.setUpOptimizer()

            def after_iteration(agent):
                trfl.update_target_variables(
                    agent.behavioral_policy.trainable_variables,
                    agent.policy.trainable_variables)

            agent = v_trace_proximal_policy_optimization_agent_impl.VTraceProximalPolicyOptimizationAgent(
                policy=self.policy, 
                behavioral_policy=self.behavioral_policy,
                value=self.value,
                optimizer=self.optimizer)
            self.assertNaiveStrategyConvergedAfter(
                agent,
                iterations=150,
                epochs=5,
                explore_episodes=128,
                explore_max_steps=200,
                exploit_episodes=10,
                exploit_max_steps=200,
                max_returns=200.,
                after_iteration=after_iteration)

    @test_util.skip_if(True)
    def testConvergenceRecurrentDiscrete(self):
        with context.eager_mode():
            self.setUpEnv(
                'CartPole-v0',
                seed=42,
                clip_inf=40.)
            self.setUpRecurrentDiscretePolicy()
            self.setUpRecurrentDiscreteBehavioralPolicy()
            self.setUpNonLinearStateValue()
            self.assignBehavioralPolicy()
            self.setUpOptimizer()

            def after_iteration(agent):
                trfl.update_target_variables(
                    agent.behavioral_policy.trainable_variables,
                    agent.policy.trainable_variables)

            agent = v_trace_proximal_policy_optimization_agent_impl.VTraceProximalPolicyOptimizationAgent(
                policy=self.policy, 
                behavioral_policy=self.behavioral_policy,
                value=self.value,
                optimizer=self.optimizer)
            self.assertNaiveStrategyConvergedAfter(
                agent,
                iterations=150,
                epochs=5,
                explore_episodes=128,
                explore_max_steps=200,
                exploit_episodes=10,
                exploit_max_steps=200,
                max_returns=200.,
                after_iteration=after_iteration)

    @test_util.skip_if(True)
    def testConvergenceContinuous(self):
        with context.eager_mode():
            self.setUpEnv(
                'Pendulum-v0',
                seed=42,
                clip_inf=40.)
            self.setUpContinuousPolicy(scale=2.)
            self.setUpContinuousBehavioralPolicy(scale=2.)
            self.setUpNonLinearStateValue()
            self.assignBehavioralPolicy()
            self.setUpOptimizer()

            def after_iteration(agent):
                trfl.update_target_variables(
                    agent.behavioral_policy.trainable_variables,
                    agent.policy.trainable_variables)

            agent = v_trace_proximal_policy_optimization_agent_impl.VTraceProximalPolicyOptimizationAgent(
                policy=self.policy, 
                behavioral_policy=self.behavioral_policy,
                value=self.value,
                optimizer=self.optimizer)
            self.assertNaiveStrategyConvergedAfter(
                agent,
                iterations=150,
                epochs=5,
                explore_episodes=128,
                explore_max_steps=200,
                exploit_episodes=10,
                exploit_max_steps=200,
                max_returns=200.,
                after_iteration=after_iteration)

    @test_util.skip_if(True)
    def testConvergenceRecurrentContinuous(self):
        with context.eager_mode():
            self.setUpEnv(
                'Pendulum-v0',
                seed=42,
                clip_inf=40.)
            self.setUpRecurrentContinuousPolicy(scale=2.)
            self.setUpRecurrentContinuousBehavioralPolicy(scale=2.)
            self.setUpNonLinearStateValue()
            self.assignBehavioralPolicy()
            self.setUpOptimizer()

            def after_iteration(agent):
                trfl.update_target_variables(
                    agent.behavioral_policy.trainable_variables,
                    agent.policy.trainable_variables)

            agent = v_trace_proximal_policy_optimization_agent_impl.VTraceProximalPolicyOptimizationAgent(
                policy=self.policy, 
                behavioral_policy=self.behavioral_policy,
                value=self.value,
                optimizer=self.optimizer)
            self.assertNaiveStrategyConvergedAfter(
                agent,
                iterations=150,
                epochs=5,
                explore_episodes=128,
                explore_max_steps=200,
                exploit_episodes=10,
                exploit_max_steps=200,
                max_returns=200.,
                after_iteration=after_iteration)


if __name__ == "__main__":
    test.main()