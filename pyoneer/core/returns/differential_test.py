import tensorflow as tf

from pyoneer.core.returns import differential


class DifferentialTest(tf.test.TestCase):

    def test_differential_returns_single_step(self):
        avg_reward = 1.
        returns = differential.differential_returns([[1., 1.]], avg_reward, steps=1)
        self.assertAllClose(returns, tf.constant([[2. - 2. * avg_reward, 1. - avg_reward]]), atol=1e-8)

    def test_differential_returns_two_step(self):
        avg_reward = 1.
        returns = differential.differential_returns([[1., 1., 1., 1.]], avg_reward, steps=2)
        self.assertAllClose(returns, tf.constant(
            [[3. - 3. * avg_reward, 3. - 3. * avg_reward, 2. - 2. * avg_reward, 1. - avg_reward]]), atol=1e-8)

    def test_differential_returns_n_step(self):
        avg_reward = 1.
        returns = differential.differential_returns([[1., 1.]], avg_reward)
        self.assertAllClose(returns, tf.constant([[2. - 2. * avg_reward, 1. - avg_reward]]), atol=1e-8)


if __name__ == '__main__':
    tf.enable_eager_execution()
    tf.test.main()
