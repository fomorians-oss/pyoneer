import tensorflow as tf
import tensorflow.contrib.eager as tfe


class Normalizer(tf.keras.layers.Layer):
    """
    Input normalization layer.

    Normalizes the inputs by subtracting a mean and dividing by a standard
    deviation.

    Args:
        inputs: A tensor.

    Returns:
        The normalized input tensor.
    """

    def __init__(self, loc, scale):
        super(Normalizer, self).__init__()
        self.loc = tfe.Variable(loc, trainable=False)
        self.scale = tfe.Variable(scale, trainable=False)

    def call(self, inputs):
        outputs = (
            inputs[..., None] - self.loc[None, ...]) / self.scale[None, ...]
        outputs = tf.check_numerics(outputs, 'outputs')
        return outputs


class OneHotEncoder(tf.keras.layers.Layer):
    """
    One-hot encoding layer.

    Encodes the integer inputs as one-hot vectors.

    Args:
        inputs: An integer tensor.

    Returns:
        The one-hot encoded inputs.
    """

    def __init__(self, depth):
        super(OneHotEncoder, self).__init__()
        self.depth = depth

    def call(self, inputs):
        outputs = tf.one_hot(inputs, self.depth)
        outputs = tf.check_numerics(outputs, 'outputs')
        return outputs


class Featurizer(tf.keras.layers.Layer):
    """
    Feature dictionary pre-processor.

    Pre-processes a dictionary of features by passing each feature
    through a layer and concatenating the outputs.

    Args:
        feature_layers: A dictionary of keys mapping to a Keras-compatible
                        layer.

    Returns:
        The outputs of each feature layer, concatenated in the order of the
        sorted keys.
    """

    def __init__(self, feature_layers):
        super(Featurizer, self).__init__()
        self.feature_layers = feature_layers

    def call(self, features):
        outputs_list = []
        for feature_key, feature_layer in sorted(self.feature_layers.items()):
            inputs = features[feature_key]
            outputs = feature_layer(inputs)
            outputs_list.append(outputs)
        outputs = tf.concat(outputs_list, axis=-1)
        outputs = tf.check_numerics(outputs, 'outputs')
        return outputs
