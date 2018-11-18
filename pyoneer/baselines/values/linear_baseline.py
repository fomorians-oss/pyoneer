import tensorflow as tf


class LinearBaseline(tf.layers.Layer):

    """An unbiased linear approximator.
    
    Notes:
        The intention is to be used as a baseline to reduce variance without introducing bias.
            >>> A(s, a) = Q(S, a) - b(S)
            >>> b(s) = S@W, where ~= |R - b(s)|^2
    """

    def __init__(self, l2_regularizer=1e-5):
        super(LinearBaseline, self).__init__()
        self.l2_regularizer = l2_regularizer

    def build(self, input_shape):
        self.linear = self.add_variable(
            name='linear',
            shape=[input_shape[-1], 1],
            initializer=tf.initializers.zeros(), 
            dtype=tf.float32,
            trainable=False)

    def call(self, states, returns, training=False):
        k = states.shape[0]
        states = tf.reshape(states, [-1, states.shape[-1]])

        if not training:
            baseline = tf.matmul(states, self.linear)
        else:
            returns = tf.reshape(returns, [-1, 1])
            outputs = tf.linalg.lstsq(states, returns, l2_regularizer=self.l2_regularizer)
            with tf.control_dependencies([tf.assign(self.linear, outputs)]):
                baseline = tf.matmul(states, self.linear)
        return tf.reshape(baseline, [k, -1])