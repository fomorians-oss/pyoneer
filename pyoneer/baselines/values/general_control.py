import tensorflow as tf


class ValueFunction(tf.keras.Model):

    def __init__(self, 
                 states_normalizer, 
                 layers=[64, 64], 
                 activation=tf.nn.relu):
        super(ValueFunction, self).__init__()
        kernel_initializer = tf.initializers.variance_scaling(scale=2.0)
        bias_initializer = tf.initializers.zeros()
        outputs_initializer = tf.initializers.variance_scaling(scale=2.0)
        self._hidden_layers = []
        for layer in layers:
            self._hidden_layers.append(
                tf.layers.Dense(
                    units=layer,
                    activation=activation,
                    bias_initializer=bias_initializer,
                    kernel_initializer=kernel_initializer))
        self.value = tf.keras.layers.Dense(
            units=1,
            activation=None,
            bias_initializer=bias_initializer,
            kernel_initializer=outputs_initializer)
        self.states_normalizer = states_normalizer

    def call(self, states, training=False, reset_state=True):
        states_norm = self.states_normalizer(states)
        hidden = states_norm
        for layer in self._hidden_layers:
            hidden = layer(hidden)
        value = self.value(hidden)
        return value
