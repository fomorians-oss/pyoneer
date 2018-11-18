import tensorflow as tf

from pyoneer.core.returns import discounted


class DiscountedTest(tf.test.TestCase):

    def test_discounted_returns_two_step(self):
        discount = .5
        returns = discounted.discounted_returns([[1., 1.]], discount, steps=2)
        self.assertAllClose(returns, tf.constant([[1. + .5 * 1., 1.]]))

    def test_discounted_returns_n_step(self):
        discount = .5
        returns = discounted.discounted_returns([[1., 1.]], discount)
        self.assertAllClose(returns, tf.constant([[1. + .5 * 1., 1.]]))


if __name__ == '__main__':
    tf.enable_eager_execution()
    tf.test.main()

