import tensorflow as tf
import tensorflow.contrib.eager as tfe
import tensorflow_probability as tfp


class MultiVariateContinuousControlPolicy(tf.keras.Model):

    def __init__(self, 
                 state_normalizer, 
                 action_normalizer, 
                 layers=[64, 64],
                 init_stdev=1., 
                 activation=tf.nn.relu):
        super(MultiVariateContinuousControlPolicy, self).__init__()
        self.output_size = action_normalizer.output_size

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
        self.loc = tf.layers.Dense(
            units=self.output_size,
            activation=None,
            bias_initializer=bias_initializer,
            kernel_initializer=outputs_initializer)
        self.scale = tfe.Variable(
            initial_value=tf.log(tf.exp(tf.ones(self.output_size) * init_stdev) - 1.))
        
        self.states_normalizer = state_normalizer
        self.action_normalizer = action_normalizer

    @property
    def regularization_loss(self):
        return tf.constant(0.)

    def call(self, inputs, exploring=False, training=False, reset_state=True):
        inputs = self.states_normalizer(inputs)
        hidden = inputs
        for layer in self._hidden_layers:
            hidden = layer(hidden)
        mean = self.loc(hidden)
        scale = tf.nn.softplus(self.scale)
        dist = tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=scale)
        return dist
