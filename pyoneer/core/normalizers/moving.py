import tensorflow as tf
import tensorflow.contrib.eager as tfe


class MovingNormalizer(tf.keras.Model):

    def __init__(self, shape, center=True, scale=True):
        super(MovingNormalizer, self).__init__()

        self.center = center
        self.scale = scale

        self.count = tfe.Variable(0., dtype=tf.float32, trainable=False)
        self.mean = tfe.Variable(
            tf.zeros(shape=shape, dtype=tf.float32), trainable=False)
        self.var_sum = tfe.Variable(
            tf.zeros(shape=shape, dtype=tf.float32), trainable=False)

    @property
    def std(self):
        return tf.sqrt(tf.maximum(self.var_sum / self.count, 0))

    def call(self, inputs, training=False):
        """Compute the normalization of the inputs.

        Args:
            inputs: inversely normalized input `Tensor`.
            training: bool if the mean and stddev should be updated.

        Returns:
            normalized inputs.
        """
        inputs = tf.convert_to_tensor(inputs)

        if training:
            input_shape = inputs.get_shape().as_list()
            input_dims = len(input_shape)
            mean_shape = self.mean.get_shape().as_list()
            mean_dims = len(mean_shape)

            reduce_axes = list(range(input_dims - mean_dims + 1))
            reduce_shape = input_shape[:-mean_dims]

            assert input_dims > mean_dims
            count = tf.cast(tf.reduce_prod(reduce_shape), tf.float32)
            self.count.assign_add(count)

            mean_deltas = tf.reduce_sum(
                inputs - self.mean, axis=reduce_axes)
            new_mean = tf.where(
                tf.greater(self.count, 1.),
                self.mean + (mean_deltas / tf.to_float(self.count)),
                inputs[tuple([0] * (len(inputs.shape) - 1))])

            var_deltas = (inputs - self.mean) * (inputs - new_mean)
            new_var_sum = self.var_sum + tf.reduce_sum(var_deltas, axis=reduce_axes)
            self.mean.assign(new_mean)
            self.var_sum.assign(new_var_sum)

        if self.center:
            inputs -= self.mean

        if self.scale:
            std = tf.where(tf.abs(self.std) <= 1e-6, tf.ones_like(self.std), self.std)
            inputs /= std

        inputs = tf.check_numerics(inputs, 'inputs (post-normalization)')
        return inputs

    def inverse(self, inputs):
        """Compute the inverse normalization of the inputs.

        Args:
        inputs: normalized input `Tensor`.

        Returns:
        inversely normalized inputs.
        """
        inputs = tf.convert_to_tensor(inputs)

        if self.scale:
            std = tf.where(abs(self.std) <= 1e-6, tf.ones_like(self.std), self.std)
            inputs *= std

        if self.center:
            inputs += self.mean

        inputs = tf.check_numerics(inputs, 'inputs (post-normalization)')
        return inputs
