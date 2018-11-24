import tensorflow as tf

from tensorflow.python.eager import context
from tensorflow.python.platform import test

from pyoneer.layers.features import Normalizer, OneHotEncoder, Featurizer


class FeatureLayersTest(test.TestCase):
    def test_normalizer_layer(self):
        with context.eager_mode():
            layer = Normalizer(loc=0.5, scale=2.0)
            inputs = tf.constant([1.0])
            outputs = layer(inputs)
            expected = tf.constant([[0.25]])
            self.assertAllEqual(outputs, expected)

    def test_one_hot_layer(self):
        with context.eager_mode():
            layer = OneHotEncoder(depth=4)
            inputs = tf.constant([3])
            outputs = layer(inputs)
            expected = tf.constant([[0.0, 0.0, 0.0, 1.0]])
            self.assertAllEqual(outputs, expected)

    def test_features_layer(self):
        with context.eager_mode():
            layer = Featurizer({
                'categorical': OneHotEncoder(depth=4),
                'scalar': Normalizer(loc=0.5, scale=2.0),
            })
            features = {
                'categorical': tf.constant([3]),
                'scalar': tf.constant([1.0]),
            }
            outputs = layer(features)
            expected = tf.constant([[0.0, 0.0, 0.0, 1.0, 0.25]])
            self.assertAllEqual(outputs, expected)


if __name__ == '__main__':
    test.main()
