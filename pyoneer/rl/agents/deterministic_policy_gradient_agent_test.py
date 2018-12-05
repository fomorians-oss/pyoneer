from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.platform import test
from tensorflow.python.framework import test_util

import trfl

from pyoneer.rl.agents import deterministic_policy_gradient_agent_impl
from pyoneer.rl.agents.test import action_value_test_case_impl


class DeterministicPolicyGradientAgentTest(action_value_test_case_impl.ActionValueTestCase):

    # @test_util.skip_if(True)
    def testConvergenceContinuous(self):
        with context.eager_mode():
            self.setUpEnv(
                'Pendulum-v0',
                seed=42,
                clip_inf=40.)
            self.setUpScalePolicy()
            self.setUpTargetScalePolicy()
            self.setUpNonLinearActionValue()
            self.setUpTargetNonLinearActionValue()
            self.setUpPolicyOptimizer()
            self.setUpValueOptimizer()

            def after_iteration(agent):
                trfl.update_target_variables(
                    agent.target_value.trainable_variables + agent.target_policy.trainable_variables,
                    agent.value.trainable_variables + agent.policy.trainable_variables)

            agent = deterministic_policy_gradient_agent_impl.DeterministicPolicyGradientAgent(
                policy=self.policy,
                target_policy=self.target_policy,
                value=self.value, 
                target_value=self.target_value,
                policy_optimizer=self.policy_optimizer,
                value_optimizer=self.value_optimizer)
            self.assertNaiveScalePolicyStrategyConvergedAfter(
                agent,
                iterations=100,
                epochs=10,
                batch_size=128,
                replay_size=128*16,
                explore_episodes=128,
                explore_max_steps=250,
                exploit_episodes=10,
                exploit_max_steps=250,
                max_returns=-200.,
                after_iteration=after_iteration)


if __name__ == "__main__":
    test.main()